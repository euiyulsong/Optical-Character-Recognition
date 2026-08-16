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
| 1 | 10.4009 | 1.0000 | 0.0000 |
| 2 | 4.2095 | 1.0000 | 0.0000 |
| 3 | 4.0276 | 1.0000 | 0.0000 |
| 4 | 3.9725 | 1.0000 | 0.0000 |
| 5 | **3.9215** | **1.0000** | **0.0000** |

### Result

- Loss: **10.4009 → 3.9215** (**62.3% ↓**)
- CER: **1.0000**
- Exact Match: **0.0000**
- Training loss decreased substantially, but recognition quality did not improve.
- Further investigation is required for label encoding, CTC decoding, vocabulary mapping, data size, and training duration.

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
- The result indicates that the model learned from fine-tuning, but has not yet reached usable OCR quality.

---

# Overall Summary

| Task | Model | Main Metric | Result |
|---|---|---|---|
| Detection | Mini DBNet | Pixel IoU ↑ | **0.5781 best** |
| Recognition | SVTR-LCNet | CER ↓ / EM ↑ | **1.0000 / 0.0000** |
| Unwarping | Mini UVDoc | L1 ↓ | **10.87% improvement** |
| Unwarping | Mini UVDoc | PSNR ↑ | **+0.655 dB** |
| Unwarping | Mini UVDoc | SSIM ↑ | **+0.0385** |
| End-to-End OCR | MinerU2.5 | CER ↓ | **14.45% improvement** |
| End-to-End OCR | MinerU2.5 | Similarity ↑ | **10.39% improvement** |

## Conclusion

- **Detection:** Mini DBNet successfully learned text-region detection, reaching a best Pixel IoU of **0.5781**.
- **Recognition:** SVTR-LCNet training loss decreased by **62.3%**, but CER and EM showed no improvement, indicating that the recognition setup requires further investigation.
- **Unwarping:** Mini UVDoc successfully improved all reconstruction metrics, with **10.87% lower L1**, **+0.655 dB PSNR**, and **+0.0385 SSIM**.
- **End-to-End OCR:** MinerU2.5 fine-tuning reduced CER by **14.45%** and improved similarity by **10.39%**, but absolute OCR accuracy remained poor.
