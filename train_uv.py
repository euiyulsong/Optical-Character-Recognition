import json
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision import models
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 384

# UVDoc-style sparse grid
GRID_H = 16
GRID_W = 16

TRAIN_N = 1000
VAL_N = 200

BATCH_SIZE = 4

EPOCHS = 5

LR = 1e-4


# ============================================================
# Base grid
#
# normalized coordinates [-1, 1]
# ============================================================

def make_base_grid(h, w):

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij",
    )

    grid = torch.stack(
        [xx, yy],
        dim=-1,
    )

    return grid


# ============================================================
# Synthetic document warp
# ============================================================

def create_random_warp(size):

    """
    clean image를 휘게 만들기 위한 backward sampling grid.

    실제 UVDoc dataset의 physical deformation 대신
    작은 실험에서는 smooth random deformation 사용.
    """

    base = make_base_grid(
        size,
        size
    )

    # -----------------------------------------------
    # coarse random displacement
    # -----------------------------------------------

    control = torch.randn(
        1,
        2,
        6,
        6,
    )

    # deformation 정도
    strength = random.uniform(
        0.03,
        0.15
    )

    control *= strength

    flow = F.interpolate(
        control,
        size=(size, size),
        mode="bicubic",
        align_corners=True,
    )

    flow = flow[
        0
    ].permute(
        1,
        2,
        0
    )

    # -----------------------------------------------
    # global perspective-ish distortion
    # -----------------------------------------------

    y = base[..., 1]

    bend = random.uniform(
        -0.12,
        0.12
    )

    curve = bend * (
        1 - y ** 2
    )

    flow[..., 0] += curve

    warped_grid = (
        base + flow
    )

    warped_grid = torch.clamp(
        warped_grid,
        -1,
        1
    )

    return warped_grid


# ============================================================
# Dataset
# ============================================================

class UVDocSynthDataset(Dataset):

    def __init__(
        self,
        annotations,
        ids,
    ):

        self.items = [
            annotations[i]
            for i in ids
        ]

    def __len__(self):

        return len(
            self.items
        )

    def __getitem__(
        self,
        idx
    ):

        item = self.items[idx]

        # -----------------------------------------------
        # Clean / flat document
        # -----------------------------------------------

        image = Image.open(
            item["image"]
        ).convert("RGB")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        clean = TF.to_tensor(
            image
        )

        # -----------------------------------------------
        # generate warp
        # -----------------------------------------------

        warp_grid = create_random_warp(
            IMAGE_SIZE
        )

        warped = F.grid_sample(
            clean.unsqueeze(0),

            warp_grid.unsqueeze(0),

            mode="bilinear",

            padding_mode="border",

            align_corners=True,
        )[0]

        # ------------------------------------------------
        # Training GT:
        #
        # low-resolution backward grid
        # ------------------------------------------------

        gt_grid = warp_grid.permute(
            2,
            0,
            1
        ).unsqueeze(0)

        gt_grid = F.interpolate(
            gt_grid,

            size=(
                GRID_H,
                GRID_W
            ),

            mode="bilinear",

            align_corners=True,
        )[0]

        return (
            warped,
            clean,
            gt_grid
        )


# ============================================================
# Mini UVDoc network
# ============================================================

class MiniUVDoc(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.encoder = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # feature:
        # B x 512 x ~12 x 12

        self.grid_head = nn.Sequential(

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (
                    GRID_H,
                    GRID_W
                )
            ),

            # x / y coordinate
            nn.Conv2d(
                128,
                2,
                kernel_size=1
            )
        )

    def forward(
        self,
        x
    ):

        feat = self.encoder(
            x
        )

        displacement = self.grid_head(
            feat
        )

        # keep predicted distortion bounded
        displacement = torch.tanh(
            displacement
        ) * 0.3

        # -----------------------------------------------
        # identity grid
        # -----------------------------------------------

        base = make_base_grid(
            GRID_H,
            GRID_W
        ).to(
            x.device
        )

        base = base.permute(
            2,
            0,
            1
        )

        base = base.unsqueeze(
            0
        ).repeat(
            x.shape[0],
            1,
            1,
            1
        )

        grid = (
            base
            + displacement
        )

        return grid


# ============================================================
# Grid → reconstructed image
# ============================================================

def reconstruct(
    warped,
    low_grid
):

    B = warped.shape[0]

    # B 2 H W
    grid = F.interpolate(
        low_grid,

        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        mode="bicubic",

        align_corners=True
    )

    # grid_sample wants:
    #
    # B H W 2

    grid = grid.permute(
        0,
        2,
        3,
        1
    )

    grid = torch.clamp(
        grid,
        -1,
        1
    )

    output = F.grid_sample(

        warped,

        grid,

        mode="bilinear",

        padding_mode="border",

        align_corners=True,
    )

    return output


# ============================================================
# Smoothness loss
# ============================================================

def smooth_loss(grid):

    dx = (
        grid[:, :, :, 1:]
        - grid[:, :, :, :-1]
    ).abs().mean()

    dy = (
        grid[:, :, 1:, :]
        - grid[:, :, :-1, :]
    ).abs().mean()

    return (
        dx + dy
    )


# ============================================================
# Dataset
# ============================================================

with open(
    "data/annotations.json",
    encoding="utf-8"
) as f:

    annotations = json.load(
        f
    )


print(
    "annotations:",
    len(annotations)
)


train_ds = UVDocSynthDataset(

    annotations,

    range(
        min(
            TRAIN_N,
            len(annotations)
        )
    ),
)


val_start = min(
    TRAIN_N,
    len(annotations)
)


val_end = min(
    TRAIN_N + VAL_N,
    len(annotations)
)


val_ds = UVDocSynthDataset(

    annotations,

    range(
        val_start,
        val_end
    )
)


train_loader = DataLoader(

    train_ds,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    pin_memory=True,
)


val_loader = DataLoader(

    val_ds,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True,
)


# ============================================================
# Model
# ============================================================

model = MiniUVDoc().to(
    DEVICE
)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=1e-4
)


# ============================================================
# Training
# ============================================================

Path(
    "output"
).mkdir(
    exist_ok=True
)


for epoch in range(
    EPOCHS
):

    model.train()

    losses = []

    grid_losses = []

    image_losses = []

    for (
        warped,
        clean,
        gt_grid
    ) in train_loader:

        warped = warped.to(
            DEVICE,
            non_blocking=True
        )

        clean = clean.to(
            DEVICE,
            non_blocking=True
        )

        gt_grid = gt_grid.to(
            DEVICE,
            non_blocking=True
        )

        # -------------------------------------------
        # predict sparse backward grid
        # -------------------------------------------

        pred_grid = model(
            warped
        )

        # -------------------------------------------
        # reconstruct flat document
        # -------------------------------------------

        reconstructed = reconstruct(
            warped,
            pred_grid
        )

        # -------------------------------------------
        # Loss 1: grid regression
        # -------------------------------------------

        loss_grid = F.smooth_l1_loss(

            pred_grid,

            gt_grid
        )

        # -------------------------------------------
        # Loss 2: image reconstruction
        # -------------------------------------------

        loss_image = F.l1_loss(

            reconstructed,

            clean
        )

        # -------------------------------------------
        # Loss 3: grid smoothness
        # -------------------------------------------

        loss_smooth = smooth_loss(
            pred_grid
        )

        # -------------------------------------------
        # combined
        # -------------------------------------------

        loss = (

            loss_grid

            + 0.5 * loss_image

            + 0.01 * loss_smooth
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(
            loss.item()
        )

        grid_losses.append(
            loss_grid.item()
        )

        image_losses.append(
            loss_image.item()
        )


    # ========================================================
    # validation
    # ========================================================

    model.eval()

    val_losses = []


    with torch.no_grad():

        for (
            warped,
            clean,
            gt_grid
        ) in val_loader:

            warped = warped.to(
                DEVICE
            )

            clean = clean.to(
                DEVICE
            )

            gt_grid = gt_grid.to(
                DEVICE
            )

            pred_grid = model(
                warped
            )

            reconstructed = reconstruct(
                warped,
                pred_grid
            )

            loss = F.l1_loss(
                reconstructed,
                clean
            )

            val_losses.append(
                loss.item()
            )


    print(
        f"epoch {epoch+1}/{EPOCHS} | "
        f"loss={np.mean(losses):.5f} | "
        f"grid={np.mean(grid_losses):.5f} | "
        f"image={np.mean(image_losses):.5f} | "
        f"val={np.mean(val_losses):.5f}"
    )


    torch.save(

        model.state_dict(),

        "output/uvdoc_small.pt"
    )


print()
print(
    "saved: output/uvdoc_small.pt"
)
