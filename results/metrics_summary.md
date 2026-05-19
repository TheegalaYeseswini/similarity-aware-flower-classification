# Metrics Summary

The main archived results currently available in the repository are:

| Method | Validation Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cross Entropy baseline | Not archived | 88.03% | 86.43% | 89.41% | 87.24% | 87.97% |
| Adaptive entropy-aware loss | 94.12% | 90.40% | 89.28% | 92.12% | 90.17% | 90.36% |
| Similarity-aware adaptive loss | 92.75% | 90.45% | 90.27% | 91.85% | 90.49% | 90.43% |

## Notes
- The Cross Entropy baseline test metrics were recovered from the archived confusion matrix.
- The original baseline validation summary was not persisted as a JSON artifact in the current repository.
- The similarity-aware model delivered the strongest held-out test accuracy among the archived experiments.
