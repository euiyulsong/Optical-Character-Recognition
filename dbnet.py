import json
import math
from pathlib import Path

import cv2
import numpy as np
import pyclipper

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF


# =========================================================
# config
# =========================================================
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4

SHRINK_RATIO = 0.4
DB_K = 50.0

TRAIN_N = 1000
VAL_N = 200


# =========================================================
# annotation parsing
# =========================================================
def get_word_boxes(item):

    boxes = item["boxes"]

    result = []

    for box in boxes:

        arr = np.asarray(
            box,
            dtype=np.float32
        )

        if arr.shape == (2, 4):
            arr = arr.T

        if arr.shape != (4, 2):
            continue

        result.append(arr)

    return result
# =========================================================
# polygon shrink
# =========================================================

def polygon_area(poly):

    return abs(
        cv2.contourArea(
            poly.astype(np.float32)
        )
    )


def polygon_perimeter(poly):

    return cv2.arcLength(
        poly.astype(np.float32),
        True,
    )


def shrink_polygon(poly, ratio=0.4):

    area = polygon_area(poly)
    perimeter = polygon_perimeter(poly)

    if perimeter <= 0:
        return None

    # DBNet shrink distance
    distance = (
        area
        * (1 - ratio ** 2)
        / perimeter
    )

    pco = pyclipper.PyclipperOffset()

    path = [
        (int(x), int(y))
        for x, y in poly
    ]

    pco.AddPath(
        path,
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )

    shrunk = pco.Execute(
        -distance
    )

    if len(shrunk) == 0:
        return None

    shrunk = max(
        shrunk,
        key=lambda x: abs(
            cv2.contourArea(
                np.asarray(x)
            )
        )
    )

    return np.asarray(
        shrunk,
        dtype=np.int32
    )


# =========================================================
# threshold map
# =========================================================

def make_threshold_map(
    polygons,
    h,
    w,
):

    """
    Quick DBNet-style threshold supervision.

    Real DBNet uses distance-to-boundary values.
    여기서는 bbox/poly 주변에 0.3~0.7 범위의
    soft threshold map을 만든다.
    """

    threshold = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    threshold_mask = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for poly in polygons:

        poly = poly.astype(np.int32)

        area = polygon_area(poly)
        perimeter = polygon_perimeter(poly)

        if perimeter <= 0:
            continue

        distance = (
            area
            * (1 - SHRINK_RATIO ** 2)
            / perimeter
        )

        pco = pyclipper.PyclipperOffset()

        pco.AddPath(
            poly.tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )

        expanded = pco.Execute(
            distance
        )

        if not expanded:
            continue

        expanded = np.asarray(
            expanded[0],
            dtype=np.int32,
        )

        cv2.fillPoly(
            threshold_mask,
            [expanded],
            1.0,
        )

        # distance transform 기반
        temp = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            temp,
            [expanded],
            255,
        )

        cv2.fillPoly(
            temp,
            [poly],
            0,
        )

        dist = cv2.distanceTransform(
            temp,
            cv2.DIST_L2,
            5,
        )

        if dist.max() > 0:

            dist = dist / dist.max()

        # boundary 근처 낮고 바깥쪽 높게
        value = 0.3 + 0.4 * dist

        threshold = np.maximum(
            threshold,
            value,
        )

    return (
        threshold,
        threshold_mask
    )


# =========================================================
# Dataset
# =========================================================

class SynthTextDBDataset(Dataset):

    def __init__(
        self,
        annotations,
        indices,
    ):

        self.items = [
            annotations[i]
            for i in indices
        ]

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]

        image = Image.open(
            item["image"]
        ).convert("RGB")

        orig_w, orig_h = image.size

        polygons = get_word_boxes(
            item
        )

        # -----------------------------------------
        # resize polygon coordinates
        # -----------------------------------------

        sx = IMAGE_SIZE / orig_w
        sy = IMAGE_SIZE / orig_h

        resized_polygons = []

        for poly in polygons:

            p = poly.copy()

            p[:, 0] *= sx
            p[:, 1] *= sy

            resized_polygons.append(p)

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # -----------------------------------------
        # probability GT
        # -----------------------------------------

        prob_gt = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float32,
        )

        for poly in resized_polygons:

            shrunk = shrink_polygon(
                poly,
                SHRINK_RATIO,
            )

            if shrunk is None:
                continue

            cv2.fillPoly(
                prob_gt,
                [shrunk],
                1.0,
            )

        # -----------------------------------------
        # threshold GT
        # -----------------------------------------

        thresh_gt, thresh_mask = (
            make_threshold_map(
                resized_polygons,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        # -----------------------------------------
        # image tensor
        # -----------------------------------------

        image = TF.to_tensor(
            image
        )

        # ImageNet normalization
        image = TF.normalize(
            image,
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        )

        prob_gt = torch.from_numpy(
            prob_gt
        ).unsqueeze(0)

        thresh_gt = torch.from_numpy(
            thresh_gt
        ).unsqueeze(0)

        thresh_mask = torch.from_numpy(
            thresh_mask
        ).unsqueeze(0)

        return (
            image,
            prob_gt,
            thresh_gt,
            thresh_mask,
        )


# =========================================================
# DBNet backbone
# =========================================================

class DBNet(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # channels:
        # c2 = 64
        # c3 = 128
        # c4 = 256
        # c5 = 512

        # lateral projection
        self.in2 = nn.Conv2d(
            64,
            64,
            1
        )

        self.in3 = nn.Conv2d(
            128,
            64,
            1
        )

        self.in4 = nn.Conv2d(
            256,
            64,
            1
        )

        self.in5 = nn.Conv2d(
            512,
            64,
            1
        )

        # output convs
        self.out2 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

        self.out3 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

        self.out4 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

        self.out5 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

        # merged FPN = 256 channels
        self.prob_head = self._make_head(
            256
        )

        self.thresh_head = self._make_head(
            256
        )

    def _make_head(self, cin):

        return nn.Sequential(

            nn.Conv2d(
                cin,
                64,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                64,
                2,
                stride=2,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                1,
                2,
                stride=2,
            ),

        )

    def forward(self, x):

        # -----------------------------------------
        # ResNet backbone
        # -----------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # -----------------------------------------
        # FPN
        # -----------------------------------------

        p5 = self.in5(c5)

        p4 = (
            self.in4(c4)
            + F.interpolate(
                p5,
                size=c4.shape[-2:],
                mode="nearest",
            )
        )

        p3 = (
            self.in3(c3)
            + F.interpolate(
                p4,
                size=c3.shape[-2:],
                mode="nearest",
            )
        )

        p2 = (
            self.in2(c2)
            + F.interpolate(
                p3,
                size=c2.shape[-2:],
                mode="nearest",
            )
        )

        p2 = self.out2(p2)
        p3 = self.out3(p3)
        p4 = self.out4(p4)
        p5 = self.out5(p5)

        target_size = p2.shape[-2:]

        p3 = F.interpolate(
            p3,
            target_size,
            mode="bilinear",
            align_corners=False,
        )

        p4 = F.interpolate(
            p4,
            target_size,
            mode="bilinear",
            align_corners=False,
        )

        p5 = F.interpolate(
            p5,
            target_size,
            mode="bilinear",
            align_corners=False,
        )

        fuse = torch.cat(
            [p2, p3, p4, p5],
            dim=1,
        )

        # -----------------------------------------
        # heads
        # -----------------------------------------

        prob_logits = self.prob_head(
            fuse
        )

        thresh_logits = self.thresh_head(
            fuse
        )

        # force to input size
        prob_logits = F.interpolate(
            prob_logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        thresh_logits = F.interpolate(
            thresh_logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        prob = torch.sigmoid(
            prob_logits
        )

        thresh = torch.sigmoid(
            thresh_logits
        )

        # DB
        binary = torch.sigmoid(
            DB_K * (
                prob - thresh
            )
        )

        return {
            "prob_logits": prob_logits,
            "thresh_logits": thresh_logits,
            "prob": prob,
            "thresh": thresh,
            "binary": binary,
        }


# =========================================================
# loss
# =========================================================

def dice_loss(pred, target):

    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (
        pred * target
    ).sum(dim=1)

    union = (
        pred.sum(dim=1)
        + target.sum(dim=1)
    )

    dice = (
        2 * intersection + 1
    ) / (
        union + 1
    )

    return 1 - dice.mean()


def masked_l1(
    pred,
    target,
    mask,
):

    diff = (
        torch.abs(
            pred - target
        ) * mask
    )

    return (
        diff.sum()
        / (mask.sum() + 1e-6)
    )


def db_loss(
    outputs,
    prob_gt,
    thresh_gt,
    thresh_mask,
):

    # probability segmentation loss
    prob_bce = (
        F.binary_cross_entropy_with_logits(
            outputs["prob_logits"],
            prob_gt,
        )
    )

    # DB binary map
    binary_dice = dice_loss(
        outputs["binary"],
        prob_gt,
    )

    # threshold regression
    thresh_loss = masked_l1(
        outputs["thresh"],
        thresh_gt,
        thresh_mask,
    )

    # simplified DB loss
    total = (
        prob_bce
        + binary_dice
        + 10.0 * thresh_loss
    )

    return (
        total,
        prob_bce,
        binary_dice,
        thresh_loss,
    )


# =========================================================
# metrics
# =========================================================

def pixel_iou(
    pred,
    gt,
):

    pred = (
        pred > 0.3
    ).float()

    intersection = (
        pred * gt
    ).sum()

    union = (
        ((pred + gt) > 0)
        .float()
        .sum()
    )

    return (
        intersection
        / (union + 1e-6)
    ).item()


# =========================================================
# load annotations
# =========================================================

with open(
    "data/annotations.json",
    encoding="utf-8",
) as f:

    annotations = json.load(f)


if len(annotations) < TRAIN_N + VAL_N:

    raise RuntimeError(
        f"Need {TRAIN_N + VAL_N} images, "
        f"but got {len(annotations)}"
    )


train_ds = SynthTextDBDataset(
    annotations,
    range(TRAIN_N),
)

val_ds = SynthTextDBDataset(
    annotations,
    range(
        TRAIN_N,
        TRAIN_N + VAL_N,
    ),
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


# =========================================================
# train
# =========================================================

model = DBNet().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


Path("output").mkdir(
    exist_ok=True
)


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # train
    # -----------------------------------------------------

    model.train()

    train_losses = []

    for (
        image,
        prob_gt,
        thresh_gt,
        thresh_mask,
    ) in train_loader:

        image = image.to(
            DEVICE,
            non_blocking=True,
        )

        prob_gt = prob_gt.to(
            DEVICE,
            non_blocking=True,
        )

        thresh_gt = thresh_gt.to(
            DEVICE,
            non_blocking=True,
        )

        thresh_mask = thresh_mask.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            image
        )

        (
            loss,
            prob_loss,
            binary_loss,
            thresh_loss,
        ) = db_loss(
            outputs,
            prob_gt,
            thresh_gt,
            thresh_mask,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

    # -----------------------------------------------------
    # validation
    # -----------------------------------------------------

    model.eval()

    val_ious = []

    with torch.no_grad():

        for (
            image,
            prob_gt,
            thresh_gt,
            thresh_mask,
        ) in val_loader:

            image = image.to(
                DEVICE
            )

            prob_gt = prob_gt.to(
                DEVICE
            )

            outputs = model(
                image
            )

            iou = pixel_iou(
                outputs["prob"],
                prob_gt,
            )

            val_ious.append(
                iou
            )

    avg_loss = np.mean(
        train_losses
    )

    avg_iou = np.mean(
        val_ious
    )

    print(
        f"epoch={epoch+1}/{EPOCHS} "
        f"loss={avg_loss:.4f} "
        f"pixel_iou={avg_iou:.4f}"
    )

    torch.save(
        model.state_dict(),
        "output/dbnet_last.pt",
    )


print(
    "saved: output/dbnet_last.pt"
)

Path("results").mkdir(exist_ok=True)
with open("results/det_dbnet.json", "w", encoding="utf-8") as f:
    json.dump({
        "task": "Detection",
        "model": "Mini DBNet",
        "samples": len(val_ds),
        "pixel_iou": float(avg_iou),
        "train_loss": float(avg_loss),
    }, f, indent=2)
print("saved: results/det_dbnet.json")
