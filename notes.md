|                                                | PSNR  | MSSIM | MSE  | AGE  | pEPS  | pCEPS | Notes       |
|------------------------------------------------|-------|-------|------|------|-------|-------|-------------|
| midterm (no comp,no mask use)                  | 35.25 | 96.70 | 0.06 | 1.62 | 6.20  | 1.62  |             |
| midterm (no comp,with stroke blend)            | 35.90 | 96.47 | 0.05 | 1.36 | 6.41  | 1.55  | clamp added |
| midterm (no comp,with stroke blend + dilation) | 36.06 | 96.94 | 0.05 | 1.29 | 6.18  | 1.62  | dilation=7  |
| midterm (no comp,with mask blend)              | 36.38 | 96.99 | 0.05 | 1.28 | 5.84  | 1.50  | clamp added |
| orig, no mask use, compressed                  | 32.86 | 94.37 | 0.07 | 3.44 | 8.27  | 1.65  |             |
| orig, (comp,with mask blend)                   | 34.22 | 94.55 | 0.06 | 2.25 | 7.10  | 1.58  |             |
| orig, no mask use, no compress                 | 34.05 | 96.56 | 0.06 | 2.88 | 6.31  | 1.64  |             |
| orig, (no comp,with stroke blend)              | 26.76 | 92.11 | 0.30 | 3.00 | 33.09 | 12.69 |             |
| orig, (no comp,with mask blend)                | 36.12 | 96.94 | 0.05 | 1.37 | 6.00  | 1.55  |             |
| new (nocomp, dataaug, newloss)                 |       |       |      |      |       |       |             |
