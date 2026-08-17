# sam3_train.py
#
# Same input as download.py:
#
# data/train_raw.jsonl
# data/val_raw.jsonl
#
# {"image": "/path/data/images/000001.jpg", "text": "hello"}
#
# IMPORTANT:
# This is a quick SAM3 adaptation experiment.
# Since the OCR dataset has NO segmentation masks,
# we generate a pseudo-mask covering the whole text crop.
#
# Install SAM3 first:
#   git clone https://github.com/facebookresearch/sam3
#   cd sam3
#   pip install -e .
#
# Then run:
#   python3 sam3_train.py

import os
import json
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/train_raw.jsonl"
VAL_FILE = "data/val_raw.jsonl"

MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200

EPOCHS = 3
BATCH_SIZE = 1

LR = 1e-5

IMAGE_SIZE = 512

OUTPUT_PATH = "sam3_text_crop.pt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.backends.cudnn.enabled = False


# ============================================================
# DATASET
# ============================================================

class OCRSegDataset(Dataset):

    def __init__(
        self,
        path,
        max_samples=None,
    ):

        self.rows = []

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                row = json.loads(line)

                image_path = row.get("image")

                if image_path is None:
                    continue

                if not os.path.exists(image_path):
                    continue

                self.rows.append(row)

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

        print(
            path,
            len(self.rows),
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(
        self,
        idx,
    ):

        row = self.rows[idx]

        image = (
            Image
            .open(row["image"])
            .convert("RGB")
        )

        w, h = image.size

        # --------------------------------------------------
        # pseudo segmentation target
        #
        # 전체 text crop을 foreground로 본다.
        # --------------------------------------------------

        mask = np.ones(
            (h, w),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # resize
        # --------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        mask = (
            Image
            .fromarray(
                (mask * 255)
                .astype(np.uint8)
            )
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            )
        )

        image = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        image = torch.tensor(
            image
        ).permute(
            2,
            0,
            1,
        )

        mask = torch.tensor(
            np.asarray(mask),
            dtype=torch.float32,
        ) / 255.0

        # box prompt = entire crop
        box = torch.tensor(
            [
                0,
                0,
                IMAGE_SIZE - 1,
                IMAGE_SIZE - 1,
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "mask": mask,
            "box": box,
            "text": row.get(
                "text",
                "",
            ),
        }


# ============================================================
# LOAD SAM3
# ============================================================

def load_sam3():

    #
    # Official repo APIs may change.
    # The exact builder name can differ by checkout.
    #

    try:

        from sam3.build_sam import build_sam3

        model = build_sam3()

    except ImportError:

        from sam3.model_builder import build_sam3

        model = build_sam3()

    model.to(
        DEVICE
    )

    return model


# ============================================================
# IOU
# ============================================================

def iou_score(
    pred,
    target,
):

    pred = (
        pred > 0.5
    ).float()

    target = (
        target > 0.5
    ).float()

    inter = (
        pred
        * target
    ).sum()

    union = (
        (
            pred
            + target
        )
        > 0
    ).float().sum()

    if union == 0:

        return 1.0

    return (
        inter
        / union
    ).item()


# ============================================================
# FORWARD WRAPPER
# ============================================================

def forward_sam(
    model,
    images,
    boxes,
):

    #
    # SAM3 repository internals can vary.
    #
    # This wrapper assumes:
    #
    #   model(
    #       images=...,
    #       boxes=...
    #   )
    #
    # returns dict with mask_logits / masks.
    #
    # If your checkout uses predictor APIs,
    # only this function needs adapting.
    #

    outputs = model(
        images=images,
        boxes=boxes,
    )

    if isinstance(
        outputs,
        dict,
    ):

        if "mask_logits" in outputs:

            logits = outputs[
                "mask_logits"
            ]

        elif "masks" in outputs:

            logits = outputs[
                "masks"
            ]

        else:

            raise RuntimeError(
                f"Unknown SAM3 output keys: "
                f"{outputs.keys()}"
            )

    else:

        logits = outputs

    if logits.ndim == 5:

        logits = logits[
            :,
            0,
            0,
        ]

    elif logits.ndim == 4:

        logits = logits[
            :,
            0,
        ]

    return logits


# ============================================================
# TRAIN
# ============================================================

def train():

    model = load_sam3()

    print(
        "model:",
        model.__class__.__name__,
    )

    train_ds = OCRSegDataset(
        TRAIN_FILE,
        MAX_TRAIN_SAMPLES,
    )

    val_ds = OCRSegDataset(
        VAL_FILE,
        MAX_VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    # ========================================================
    # train
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for step, batch in enumerate(
            train_loader
        ):

            images = batch[
                "image"
            ].to(
                DEVICE
            )

            masks = batch[
                "mask"
            ].to(
                DEVICE
            )

            boxes = batch[
                "box"
            ].to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = forward_sam(
                model,
                images,
                boxes,
            )

            if (
                logits.shape[-2:]
                != masks.shape[-2:]
            ):

                logits = (
                    F.interpolate(
                        logits.unsqueeze(1),
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(1)
                )

            # BCE mask loss

            bce = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    masks,
                )
            )

            probs = (
                torch.sigmoid(
                    logits
                )
            )

            # Dice loss

            inter = (
                probs
                * masks
            ).sum()

            dice = (
                1
                -
                (
                    2 * inter + 1
                )
                /
                (
                    probs.sum()
                    + masks.sum()
                    + 1
                )
            )

            loss = (
                bce
                + dice
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            if step % 20 == 0:

                print(
                    f"epoch={epoch+1}/{EPOCHS} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        print(
            f"epoch={epoch+1} "
            f"avg_loss="
            f"{total_loss / len(train_loader):.4f}"
        )

        # ====================================================
        # validation
        # ====================================================

        model.eval()

        ious = []

        with torch.no_grad():

            for batch in val_loader:

                images = (
                    batch["image"]
                    .to(DEVICE)
                )

                masks = (
                    batch["mask"]
                    .to(DEVICE)
                )

                boxes = (
                    batch["box"]
                    .to(DEVICE)
                )

                logits = forward_sam(
                    model,
                    images,
                    boxes,
                )

                if (
                    logits.shape[-2:]
                    != masks.shape[-2:]
                ):

                    logits = (
                        F.interpolate(
                            logits.unsqueeze(1),
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(1)
                    )

                pred = (
                    torch.sigmoid(
                        logits
                    )
                )

                ious.append(
                    iou_score(
                        pred,
                        masks,
                    )
                )

        mean_iou = (
            sum(ious)
            / len(ious)
        )

        print(
            f"val IoU="
            f"{mean_iou:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    torch.save(
        model.state_dict(),
        OUTPUT_PATH,
    )

    print(
        "saved:",
        OUTPUT_PATH,
    )

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    train()
