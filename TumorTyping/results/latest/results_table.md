# Results — GSE90496 Baseline (ship-fast streamed top-k)

- Timestamp: 2026-02-19 11:27:15
- Samples: 2801
- Features used: 2000 (top-k variance; pass1 capped)
- Pass1 variance subsample: 100 samples
- Pass1 max probes scanned: 80000
- Split: train/val = 0.75/0.25 (stratified, seed=42)
- Model: StandardScaler + LinearSVC(C=1.0, class_weight=balanced)

## Metrics
| metric | value |
|---|---:|
| balanced_accuracy (val) | 0.9743 |
| macro_f1 (val) | 0.9765 |

## QC
- probes_loaded_topk: 2000
- probes_after_constant_filter: 2000

## Artifacts
- Confusion matrix: `D:/Study Material/LifeReset/Epigenetics/TumorTyping/results/latest/confusion_matrix.png`
