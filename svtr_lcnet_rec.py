import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H = 48
IMG_W = 320
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
NUM_WORKERS = 4

# Printable ASCII + CTC blank. SynthText is mainly English-oriented and this keeps the quickstart small.
CHARS = ''.join(chr(i) for i in range(32, 127))
BLANK = 0
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARS)}
ID2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NCLASS = len(CHARS) + 1


def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class RecDataset(Dataset):
    def __init__(self, path):
        self.items = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.items = [x for x in self.items if all(c in CHAR2ID for c in norm_text(x["text"])) and norm_text(x["text"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = self.items[idx]
        image = Image.open(x["image"]).convert("RGB")
        w, h = image.size
        new_w = max(1, min(IMG_W, round(w * IMG_H / max(h, 1))))
        image = image.resize((new_w, IMG_H), Image.BILINEAR)
        arr = np.asarray(image).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        canvas = torch.zeros(3, IMG_H, IMG_W, dtype=torch.float32)
        canvas[:, :, :new_w] = tensor
        text = norm_text(x["text"])
        target = torch.tensor([CHAR2ID[c] for c in text], dtype=torch.long)
        return canvas, target, text


def collate(batch):
    images, targets, texts = zip(*batch)
    lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return torch.stack(images), targets, lens, list(texts)


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class LCBlock(nn.Module):
    """LCNet-like depthwise separable convolution block."""
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = HSwish()

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        return self.act(self.pw_bn(self.pw(x)))


class SVTRBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.ffn(self.norm2(x))


class SVTRLCNet(nn.Module):
    """Small PyTorch SVTR-LCNet-style recognizer: LCNet-like CNN + SVTR sequence mixing + CTC."""
    def __init__(self, nclass):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), HSwish(),
            LCBlock(32, 64, (2, 2)),
            LCBlock(64, 96, (2, 2)),
            LCBlock(96, 128, (2, 1)),
            LCBlock(128, 192, (2, 1)),
        )
        self.svtr = nn.Sequential(SVTRBlock(192, 4), SVTRBlock(192, 4))
        self.head = nn.Linear(192, nclass)

    def forward(self, x):
        x = self.stem(x)              # B,C,H,W
        x = x.mean(dim=2).transpose(1, 2)  # B,T,C
        x = self.svtr(x)
        return self.head(x)           # B,T,C


def greedy_decode(logits):
    ids = logits.argmax(-1).cpu().tolist()
    outs = []
    for seq in ids:
        out, prev = [], None
        for i in seq:
            if i != BLANK and i != prev:
                out.append(ID2CHAR.get(i, ""))
            prev = i
        outs.append(norm_text(''.join(out)))
    return outs


train_ds = RecDataset("data/train_raw.jsonl")
val_ds = RecDataset("data/val_raw.jsonl")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

model = SVTRLCNet(NCLASS).to(DEVICE)
model.tie_weights()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
Path("output").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

best_cer = float("inf")
best = None
for epoch in range(EPOCHS):
    model.train(); losses = []
    for images, targets, target_lens, _ in train_loader:
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        logits = model(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        input_lens = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
        loss = ctc(log_probs, targets, input_lens, target_lens)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())

    model.eval(); total_ed = total_chars = exact = n = 0
    with torch.no_grad():
        for images, _, _, texts in val_loader:
            preds = greedy_decode(model(images.to(DEVICE)))
            for gt, pred in zip(texts, preds):
                total_ed += edit_distance(gt, pred); total_chars += max(len(gt), 1)
                exact += int(gt == pred); n += 1
    val_cer = total_ed / max(total_chars, 1)
    val_em = exact / max(n, 1)
    print(f"epoch={epoch+1}/{EPOCHS} loss={np.mean(losses):.4f} CER={val_cer:.4f} EM={val_em:.4f}")
    if val_cer < best_cer:
        best_cer = val_cer
        best = {"cer": val_cer, "exact_match": val_em, "epoch": epoch + 1}
        torch.save(model.state_dict(), "output/svtr_lcnet_rec_best.pt")

with open("results/rec_svtr_lcnet.json", "w", encoding="utf-8") as f:
    json.dump({"task": "Recognition", "model": "SVTR-LCNet (small)", "samples": len(val_ds), **best}, f, indent=2)
print("saved: output/svtr_lcnet_rec_best.pt")
print("saved: results/rec_svtr_lcnet.json")
