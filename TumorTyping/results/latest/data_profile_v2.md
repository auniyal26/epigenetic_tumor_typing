# Data Profile v2 — GSE90496 (Labels + Batch Scan)

- Timestamp: 2026-02-26 10:43:01
- Source file: `GSE90496_series_matrix.txt.gz`

## Labels: methylation_class
- Samples: **2801**
- Unique classes: **91**
- Class imbalance:
  - classes with <5 samples: **0**
  - classes with <10 samples: **15**
  - classes with <20 samples: **41**

### Top 15 classes by sample count

| class | n |
|---|---:|
| GBM, RTK II | 143 |
| MB, G4 | 138 |
| LGG, PA PF | 114 |
| EPN, PF A | 91 |
| MNG | 90 |
| MB, SHH CHL AD | 84 |
| O IDH | 80 |
| DMG, K27 | 78 |
| A IDH | 78 |
| MB, G3 | 77 |
| EPN, RELA | 70 |
| GBM, RTK I | 64 |
| GBM, MES | 56 |
| MB, SHH INF | 52 |
| EPN, PF B | 51 |

Plot saved: `D:/Study Material/LifeReset/Epigenetics/TumorTyping/results/latest/class_counts_top30.png`

## Batch/metadata scan (Series Matrix characteristics rows)
These are potential batch/confound fields if they exist (site, platform, tissue, processing, etc.).
We scan the first ~12 `!Sample_characteristics_ch1` rows and summarize unique values.

| row_idx | key_guess | n_unique | examples |
|---:|---|---:|---|
| 0 | methylation class | 91 | GBM, G34; DMG, K27; ATRT, SHH; CONTR, CEBM; GBM, MYCN |
| 1 | material | 2 | Frozen; FFPE |

## Suggested split policy (practical)
- Use stratified train/val split, but consider a minimum support threshold for classes (e.g., drop or group classes with <10 samples).
- Report macro metrics + per-class recall; for rare classes, metrics are unstable (high variance).

