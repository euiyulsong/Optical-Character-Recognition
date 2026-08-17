# OCR quickstart: DET + REC + UVDoc

Run everything:

```bash
bash run_all.sh
```

Pipeline:
1. Download 1,000 train + 200 val word crops for recognition.
2. Download 1,200 full SynthText samples for detection/unwarping.
3. Train Mini DBNet detection.
4. Train small PyTorch SVTR-LCNet-style CTC recognizer.
5. Train Mini UVDoc-style unwarping.
6. Evaluate UVDoc with deterministic warps.
7. Print one summary table and save JSON metrics under `results/`.

Important: the added recognizer is a compact **SVTR-LCNet-style PyTorch quickstart** (LCNet-like depthwise CNN + SVTR attention blocks + CTC), not the exact PaddleOCR production SVTR_LCNet implementation/checkpoint.
