# OCR / Document AI Mini Training Results

## 1. Text Detection — Mini DBNet

| Epoch | Loss ↓ | Pixel IoU ↑ |
|---:|---:|---:|
| 1 | 1.9057 | 0.2913 |
| 2 | 1.1542 | 0.5020 |
| 3 | 1.0252 | 0.5258 |
| 4 | 0.9342 | **0.5781** |
| 5 | **0.8758** | 0.5394 |

### Result

- Loss: **1.9057 → 0.8758** (**54.0% ↓**)
- Best Pixel IoU: **0.5781**
- Detection successfully learned meaningful text regions.
- Slight IoU degradation at Epoch 5 suggests mild overfitting or evaluation variance.

---

## 2. Text Recognition — SVTR-LCNet

| Epoch | Loss ↓ | CER ↓ | EM ↑ |
|---:|---:|---:|---:|
| 1 | 5.5162 | 1.0000 | 0.0000 |
| 2 | 3.9555 | 1.0000 | 0.0000 |
| 3 | 3.8255 | 0.9571 | 0.0100 |
| 4 | 3.6886 | 0.8976 | 0.0150 |
| 5 | 3.6238 | 0.9267 | 0.0050 |
| 6 | 3.5572 | 0.9198 | 0.0200 |
| 7 | 3.4935 | 0.9046 | 0.0300 |
| 8 | 3.4548 | 0.8935 | 0.0300 |
| 9 | 3.3868 | 0.8658 | 0.0350 |
| 10 | **3.3273** | **0.8409** | **0.0600** |

### Result

- Loss: **5.5162 → 3.3273** (**39.7% ↓**)
- CER: **1.0000 → 0.8409** (**15.9% improvement**)
- Exact Match: **0.0000 → 0.0600**
- Best Epoch: **10**
- Unlike the previous 5-epoch experiment, the model now shows clear improvement in both CER and EM.
- Recognition quality is still relatively low, but the model is successfully learning character-level recognition.

---

## 3. Document Unwarping — Mini UVDoc

| Epoch | Total Loss ↓ | Grid Loss ↓ | Image Loss ↓ | Validation Loss ↓ |
|---:|---:|---:|---:|---:|
| 1 | 0.05199 | 0.00418 | 0.09280 | 0.10513 |
| 2 | 0.04974 | 0.00458 | 0.08749 | 0.10198 |
| 3 | 0.04900 | 0.00521 | 0.08472 | 0.10133 |
| 4 | 0.04772 | 0.00529 | 0.08199 | **0.09851** |
| 5 | **0.04689** | 0.00537 | **0.08017** | 0.10114 |

### Evaluation

| Model | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Warped Baseline | 0.11214 | 14.671 dB | 0.4421 |
| **Trained UVDoc** | **0.09995** | **15.327 dB** | **0.4806** |

### Improvement

- **L1:** 10.87% improvement
- **PSNR:** +0.655 dB
- **SSIM:** +0.0385
- The trained model outperformed the warped baseline across all reconstruction metrics.

---

## 4. End-to-End Document OCR — MinerU2.5

| Model | Samples | EM ↑ | CER ↓ | Similarity ↑ |
|---|---:|---:|---:|---:|
| MinerU2.5 Base | 200 | 0.0000 | 267.0918 | 0.0077 |
| **MinerU2.5 Fine-tuned** | 200 | **0.0000** | **228.4900** | **0.0085** |

### Result

- **CER:** 267.0918 → 228.4900 (**14.45% improvement**)
- **Similarity:** 0.0077 → 0.0085 (**10.39% improvement**)
- **EM:** 0.0000 → 0.0000 (no improvement)
- Fine-tuning improved CER and similarity, but absolute OCR performance remains very low.

---

# Overall Summary

| Task | Model | Main Metric | Result |
|---|---|---|---|
| Detection | Mini DBNet | Pixel IoU ↑ | **0.5781 best** |
| Recognition | SVTR-LCNet | CER ↓ | **1.0000 → 0.8409** |
| Recognition | SVTR-LCNet | EM ↑ | **0.0000 → 0.0600** |
| Unwarping | Mini UVDoc | L1 ↓ | **10.87% improvement** |
| Unwarping | Mini UVDoc | PSNR ↑ | **+0.655 dB** |
| Unwarping | Mini UVDoc | SSIM ↑ | **+0.0385** |
| End-to-End OCR | MinerU2.5 | CER ↓ | **14.45% improvement** |
| End-to-End OCR | MinerU2.5 | Similarity ↑ | **10.39% improvement** |

## Conclusion

- **Detection:** Mini DBNet successfully learned text-region detection, reaching a best Pixel IoU of **0.5781**.
- **Recognition:** SVTR-LCNet now shows clear learning. CER improved from **1.0000 to 0.8409**, while EM increased from **0% to 6%**. Performance is still limited, but the model is successfully learning the recognition task.
- **Unwarping:** Mini UVDoc improved all reconstruction metrics, achieving **10.87% lower L1**, **+0.655 dB PSNR**, and **+0.0385 SSIM**.
- **End-to-End OCR:** MinerU2.5 fine-tuning reduced CER by **14.45%** and improved similarity by **10.39%**, although absolute OCR performance remains poor.
