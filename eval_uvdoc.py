import json
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 384

GRID_H = 16
GRID_W = 16

TRAIN_N = 1000
VAL_N = 200

BATCH_SIZE = 4


# ============================================================
# grid
# ============================================================

def make_base_grid(h, w, device=None):

    yy, xx = torch.meshgrid(
        torch.linspace(
            -1,
            1,
            h,
            device=device,
        ),
        torch.linspace(
            -1,
            1,
            w,
            device=device,
        ),
        indexing="ij",
    )

    return torch.stack(
        [xx, yy],
        dim=-1,
    )


# ============================================================
# IMPORTANT
#
# 평가에서는 항상 같은 warp를 써야 baseline/trained 비교가 가능
# ============================================================

def create_fixed_warp(
    size,
    seed,
):

    g = torch.Generator()

    g.manual_seed(seed)

    base = make_base_grid(
        size,
        size,
    )

    control = torch.randn(
        1,
        2,
        6,
        6,
        generator=g,
    )

    # sample마다 deterministic
    strength = (
        0.05
        + (seed % 100) / 1000.0
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
        0,
    )

    y = base[..., 1]

    bend = (
        ((seed % 21) - 10)
        / 100.0
    )

    curve = bend * (
        1 - y ** 2
    )

    flow[..., 0] += curve

    grid = (
        base + flow
    )

    return torch.clamp(
        grid,
        -1,
        1,
    )


# ============================================================
# dataset
# ============================================================

class EvalDataset(Dataset):

    def __init__(
        self,
        annotations,
        ids,
    ):

        self.items = [
            annotations[i]
            for i in ids
        ]

        self.ids = list(ids)

    def __len__(self):
        return len(self.items)

    def __getitem__(
        self,
        idx,
    ):

        item = self.items[idx]

        image = Image.open(
            item["image"]
        ).convert("RGB")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        clean = TF.to_tensor(
            image
        )

        # 항상 같은 distortion
        sample_id = self.ids[idx]

        warp_grid = create_fixed_warp(
            IMAGE_SIZE,
            seed=sample_id,
        )

        warped = F.grid_sample(
            clean.unsqueeze(0),
            warp_grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0]

        return (
            warped,
            clean,
        )


# ============================================================
# Mini UVDoc
# 반드시 train 코드와 동일해야 함
# ============================================================

class MiniUVDoc(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=None
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

        self.grid_head = nn.Sequential(

            nn.Conv2d(
                512,
                256,
                3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                128,
                3,
                padding=1,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (
                    GRID_H,
                    GRID_W,
                )
            ),

            nn.Conv2d(
                128,
                2,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        feat = self.encoder(x)

        displacement = (
            torch.tanh(
                self.grid_head(feat)
            )
            * 0.3
        )

        base = make_base_grid(
            GRID_H,
            GRID_W,
            x.device,
        )

        base = base.permute(
            2,
            0,
            1,
        )

        base = base.unsqueeze(
            0
        ).repeat(
            x.shape[0],
            1,
            1,
            1,
        )

        return (
            base
            + displacement
        )


# ============================================================
# reconstruction
# ============================================================

def reconstruct(
    warped,
    grid,
):

    grid = F.interpolate(
        grid,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bicubic",
        align_corners=True,
    )

    grid = grid.permute(
        0,
        2,
        3,
        1,
    )

    grid = torch.clamp(
        grid,
        -1,
        1,
    )

    return F.grid_sample(
        warped,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ============================================================
# metrics
# ============================================================

def batch_l1(
    pred,
    target,
):

    return F.l1_loss(
        pred,
        target,
        reduction="none",
    ).mean(
        dim=(1, 2, 3)
    )


def batch_psnr(
    pred,
    target,
):

    mse = (
        (pred - target) ** 2
    ).mean(
        dim=(1, 2, 3)
    )

    return (
        10
        * torch.log10(
            1.0 / (mse + 1e-8)
        )
    )


# 간단 SSIM implementation
def batch_ssim(
    x,
    y,
):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        x,
        11,
        1,
        5,
    )

    mu_y = F.avg_pool2d(
        y,
        11,
        1,
        5,
    )

    sigma_x = (
        F.avg_pool2d(
            x * x,
            11,
            1,
            5,
        )
        - mu_x ** 2
    )

    sigma_y = (
        F.avg_pool2d(
            y * y,
            11,
            1,
            5,
        )
        - mu_y ** 2
    )

    sigma_xy = (
        F.avg_pool2d(
            x * y,
            11,
            1,
            5,
        )
        - mu_x * mu_y
    )

    ssim_map = (
        (
            2 * mu_x * mu_y
            + C1
        )
        * (
            2 * sigma_xy
            + C2
        )
    ) / (
        (
            mu_x ** 2
            + mu_y ** 2
            + C1
        )
        * (
            sigma_x
            + sigma_y
            + C2
        )
    )

    return ssim_map.mean(
        dim=(1, 2, 3)
    )


# ============================================================
# data
# ============================================================

with open(
    "data/annotations.json",
    encoding="utf-8",
) as f:

    annotations = json.load(f)


val_start = TRAIN_N

val_end = min(
    TRAIN_N + VAL_N,
    len(annotations),
)


val_ds = EvalDataset(
    annotations,
    range(
        val_start,
        val_end,
    ),
)


val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)


# ============================================================
# load trained model
# ============================================================

model = MiniUVDoc().to(
    DEVICE
)

state = torch.load(
    "output/uvdoc_small.pt",
    map_location=DEVICE,
)

model.load_state_dict(
    state
)

model.eval()


# ============================================================
# evaluation
# ============================================================

baseline_l1 = []
baseline_psnr = []
baseline_ssim = []

trained_l1 = []
trained_psnr = []
trained_ssim = []


with torch.no_grad():

    for warped, clean in val_loader:

        warped = warped.to(
            DEVICE
        )

        clean = clean.to(
            DEVICE
        )

        # ========================================
        # baseline:
        # 아무것도 안 함
        # ========================================

        base_out = warped


        # ========================================
        # trained UVDoc
        # ========================================

        pred_grid = model(
            warped
        )

        restored = reconstruct(
            warped,
            pred_grid,
        )


        # ----------------------------------------
        # baseline metrics
        # ----------------------------------------

        baseline_l1.extend(
            batch_l1(
                base_out,
                clean,
            )
            .cpu()
            .tolist()
        )

        baseline_psnr.extend(
            batch_psnr(
                base_out,
                clean,
            )
            .cpu()
            .tolist()
        )

        baseline_ssim.extend(
            batch_ssim(
                base_out,
                clean,
            )
            .cpu()
            .tolist()
        )


        # ----------------------------------------
        # trained metrics
        # ----------------------------------------

        trained_l1.extend(
            batch_l1(
                restored,
                clean,
            )
            .cpu()
            .tolist()
        )

        trained_psnr.extend(
            batch_psnr(
                restored,
                clean,
            )
            .cpu()
            .tolist()
        )

        trained_ssim.extend(
            batch_ssim(
                restored,
                clean,
            )
            .cpu()
            .tolist()
        )


# ============================================================
# results
# ============================================================

b_l1 = np.mean(
    baseline_l1
)

b_psnr = np.mean(
    baseline_psnr
)

b_ssim = np.mean(
    baseline_ssim
)


t_l1 = np.mean(
    trained_l1
)

t_psnr = np.mean(
    trained_psnr
)

t_ssim = np.mean(
    trained_ssim
)


print()
print("=" * 60)

print(
    f"{'Model':<20}"
    f"{'L1 ↓':>12}"
    f"{'PSNR ↑':>12}"
    f"{'SSIM ↑':>12}"
)

print("-" * 60)

print(
    f"{'Warped baseline':<20}"
    f"{b_l1:>12.5f}"
    f"{b_psnr:>12.3f}"
    f"{b_ssim:>12.4f}"
)

print(
    f"{'Trained UVDoc':<20}"
    f"{t_l1:>12.5f}"
    f"{t_psnr:>12.3f}"
    f"{t_ssim:>12.4f}"
)

print("-" * 60)

print(
    "L1 improvement  :",
    f"{(b_l1 - t_l1) / b_l1 * 100:.2f}%"
)

print(
    "PSNR improvement:",
    f"{t_psnr - b_psnr:+.3f} dB"
)

print(
    "SSIM improvement:",
    f"{t_ssim - b_ssim:+.4f}"
)

print("=" * 60)


from pathlib import Path
Path("results").mkdir(exist_ok=True)
with open("results/uvdoc.json", "w", encoding="utf-8") as f:
    json.dump({
        "task": "Unwarping",
        "model": "Mini UVDoc",
        "samples": len(val_ds),
        "l1": float(t_l1),
        "psnr": float(t_psnr),
        "ssim": float(t_ssim),
        "baseline_l1": float(b_l1),
        "baseline_psnr": float(b_psnr),
        "baseline_ssim": float(b_ssim),
    }, f, indent=2)
print("saved: results/uvdoc.json")
