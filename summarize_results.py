import json
from pathlib import Path
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
files = sorted(Path("results").glob("*.json"))
rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

def f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"

print("\n" + "=" * 108)
print(f"{'Task':<13}{'Model':<24}{'Samples':>9}{'Main metric':<22}{'Value':>12}{'Extra metrics'}")
print("-" * 108)
for r in rows:
    task = r.get("task", "")
    if task == "Detection":
        main, val = "Pixel IoU ↑", f(r.get("pixel_iou"))
        extra = f"train_loss={f(r.get('train_loss'))}"
    elif task == "Recognition":
        main, val = "CER ↓", f(r.get("cer"))
        extra = f"EM={f(r.get('exact_match'))}, best_epoch={r.get('epoch','-')}"
    elif task == "Unwarping":
        main, val = "PSNR ↑", f(r.get("psnr"), 3)
        extra = f"SSIM={f(r.get('ssim'))}, L1={f(r.get('l1'),5)}"
    else:
        main, val, extra = "-", "-", ""
    print(f"{task:<13}{r.get('model',''):<24}{r.get('samples','-'):>9}{main:<22}{val:>12}  {extra}")
print("=" * 108)
print("Raw JSON metrics are in ./results/")
