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
- Loss: **1.9057 → 0.8758** (**54.0% decrease**)
- Best Pixel IoU: **0.5781** (Epoch 4)
- Final Pixel IoU: **0.5394**
- Detection model successfully learned meaningful text regions.
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
- Loss: **10.4009 → 3.9215** (**62.3% decrease**)
- CER: **1.0000**
- Exact Match: **0.0000**
- Training loss decreased substantially, but recognition quality did **not improve** according to CER/EM.
- The model is optimizing the training objective but has not yet learned usable character-level decoding.
- Recognition requires further investigation (e.g. label encoding/decoding, CTC blank handling, vocabulary mapping, data amount, or additional training).

---

## 3. Document Unwarping — Mini UVDoc

| Epoch | Total Loss ↓ | Grid Loss ↓ | Image Loss ↓ | Validation Loss ↓ |
|---:|---:|---:|---:|---:|
| 1 | 0.05199 | 0.00418 | 0.09280 | 0.10513 |
| 2 | 0.04974 | 0.00458 | 0.08749 | 0.10198 |
| 3 | 0.04900 | 0.00521 | 0.08472 | 0.10133 |
| 4 | 0.04772 | 0.00529 | 0.08199 | **0.09851** |
| 5 | **0.04689** | 0.00537 | **0.08017** | 0.10114 |

### Result
- Training loss: **0.05199 → 0.04689**
- Image loss: **0.09280 → 0.08017**
- Best validation loss: **0.09851** (Epoch 4)
- Validation loss slightly increased at Epoch 5, suggesting mild overfitting.

---

## 4. UVDoc Evaluation

| Model | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Warped Baseline | 0.11214 | 14.671 dB | 0.4421 |
| **Trained UVDoc** | **0.09995** | **15.327 dB** | **0.4806** |

### Improvement

- **L1:** 10.87% improvement
- **PSNR:** +0.655 dB
- **SSIM:** +0.0385

The trained UVDoc model consistently outperformed the warped baseline across all three image reconstruction metrics.

---

# Overall Summary

| Task | Model | Main Metric | Result |
|---|---|---|---|
| Detection | Mini DBNet | Pixel IoU ↑ | **0.5781 best** |
| Recognition | SVTR-LCNet | CER ↓ / EM ↑ | **1.0000 / 0.0000** |
| Unwarping | Mini UVDoc | L1 ↓ | **10.87% improvement** |
| Unwarping | Mini UVDoc | PSNR ↑ | **+0.655 dB** |
| Unwarping | Mini UVDoc | SSIM ↑ | **+0.0385** |

## Conclusion

- **Detection:** DBNet successfully learned text-region detection, reaching a best Pixel IoU of **0.5781**.
- **Recognition:** SVTR-LCNet loss decreased significantly, but **CER/EM showed no recognition improvement**, indicating that the recognition pipeline or training setup needs further investigation.
- **Unwarping:** UVDoc showed clear improvement over the warped baseline across **L1, PSNR, and SSIM**, demonstrating successful learning even with the small training setup.
