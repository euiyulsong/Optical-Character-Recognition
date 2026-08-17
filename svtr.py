import os
import re
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from jiwer import cer as jiwer_cer

# ============================================================
# CONFIG
# ============================================================
TRAIN_FILE = "data/train_raw.jsonl"
VAL_FILE = "data/val_raw.jsonl"
OUTPUT_FILE = "output/svtr_lcnet_last.pt"
RESULT_FILE = "results/rec_svtr_lcnet.json"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200
IMG_H = 48
IMG_W = 192
D_MODEL = 192
NHEAD = 6
NUM_LAYERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = False

print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

random.seed(42)
torch.manual_seed(42)


def normalize(s):
    return re.sub(r"\s+", " ", str(s).strip())


# ============================================================
# VOCAB
# ============================================================
def load_rows(path, max_samples=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            image = x.get("image")
            text = normalize(x.get("text", ""))
            if image and text and os.path.exists(image):
                rows.append({"image": image, "text": text})
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


train_rows = load_rows(TRAIN_FILE, MAX_TRAIN_SAMPLES)
val_rows = load_rows(VAL_FILE, MAX_VAL_SAMPLES)

# Build charset from train + val so evaluation never fails on unseen symbols.
chars = sorted(set("".join(x["text"] for x in train_rows + val_rows)))
BLANK = 0
char2id = {c: i + 1 for i, c in enumerate(chars)}
id2char = {i + 1: c for i, c in enumerate(chars)}
NUM_CLASSES = len(chars) + 1

print("train:", len(train_rows))
print("val  :", len(val_rows))
print("charset:", len(chars))


# ============================================================
# DATASET
# ============================================================
class RecDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["image"]).convert("RGB")
        image = self.resize_with_pad(image)
        x = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
        x = x.permute(2, 0, 1)
        x = (x - 0.5) / 0.5
        target = torch.tensor([char2id[c] for c in row["text"]], dtype=torch.long)
        return x, target, row["text"]

    @staticmethod
    def resize_with_pad(image):
        w, h = image.size
        scale = min(IMG_W / max(w, 1), IMG_H / max(h, 1))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        image = image.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        canvas.paste(image, (0, (IMG_H - nh) // 2))
        return canvas


def collate(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets, dim=0)
    return images, targets, lengths, list(texts)


# ============================================================
# LCNet-LIKE BACKBONE
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, g=1):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(cout),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.net(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, stride, g=cin)
        self.pw = ConvBNAct(cin, cout, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class LCNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(3, 32, 3, 2),          # 48x192 -> 24x96
            DepthwiseSeparable(32, 64, 2),   # 12x48
            DepthwiseSeparable(64, 96, 2),   # 6x24
            DepthwiseSeparable(96, 128, 1),
            DepthwiseSeparable(128, D_MODEL, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# SVTR-LIKE RECOGNIZER
# ============================================================
class SVTRLCNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = LCNetBackbone()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, num_classes)

    def forward(self, x):
        x = self.backbone(x)        # B,C,H,W
        x = x.mean(dim=2)           # B,C,W
        x = x.transpose(1, 2)       # B,W,C
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x)         # B,T,C


# ============================================================
# CTC DECODE / METRICS
# ============================================================
def greedy_decode(logits):
    ids = logits.argmax(-1).cpu().tolist()
    out = []
    for seq in ids:
        prev = None
        chars_out = []
        for idx in seq:
            if idx != BLANK and idx != prev:
                chars_out.append(id2char.get(idx, ""))
            prev = idx
        out.append("".join(chars_out))
    return out


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    gts, preds = [], []
    for images, _, _, texts in loader:
        images = images.to(DEVICE)
        logits = model(images)
        batch_preds = greedy_decode(logits)
        gts.extend([normalize(x) for x in texts])
        preds.extend([normalize(x) for x in batch_preds])

    em = sum(g == p for g, p in zip(gts, preds)) / max(1, len(gts))
    score_cer = jiwer_cer(gts, preds) if gts else 0.0
    return score_cer, em, preds, gts


# ============================================================
# TRAIN
# ============================================================
train_loader = DataLoader(
    RecDataset(train_rows),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
)
val_loader = DataLoader(
    RecDataset(val_rows),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate,
)

model = SVTRLCNet(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

best_cer = float("inf")
best_epoch = 0

print("\n" + "=" * 70)
print("Train recognition - SVTR-LCNet")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for images, targets, target_lengths, _ in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        logits = model(images)                # B,T,C
        log_probs = F.log_softmax(logits, -1).transpose(0, 1)  # T,B,C
        input_lengths = torch.full(
            (images.size(0),),
            log_probs.size(0),
            dtype=torch.long,
            device=DEVICE,
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    val_cer, val_em, _, _ = evaluate(model, val_loader)
    avg_loss = total_loss / max(1, len(train_loader))
    print(f"epoch={epoch}/{EPOCHS} loss={avg_loss:.4f} CER={val_cer:.4f} EM={val_em:.4f}")

    if val_cer < best_cer:
        best_cer = val_cer
        best_epoch = epoch
        os.makedirs("output", exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "char2id": char2id,
                "id2char": id2char,
                "blank": BLANK,
                "config": {
                    "img_h": IMG_H,
                    "img_w": IMG_W,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                },
            },
            OUTPUT_FILE,
        )

# Load best checkpoint for final metrics.
ckpt = torch.load(OUTPUT_FILE, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
final_cer, final_em, preds, gts = evaluate(model, val_loader)

print("\n" + "=" * 70)
print("FINAL SVTR-LCNet")
print("=" * 70)
print("best epoch:", best_epoch)
print("CER       :", final_cer)
print("EM        :", final_em)

os.makedirs("results", exist_ok=True)
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "SVTR-LCNet (compact PyTorch)",
            "samples": len(val_rows),
            "CER": final_cer,
            "EM": final_em,
            "best_epoch": best_epoch,
            "predictions": [
                {"ground_truth": g, "prediction": p}
                for g, p in zip(gts, preds)
            ],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("saved:", OUTPUT_FILE)
print("saved:", RESULT_FILE)
