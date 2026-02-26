# Results — GSE90496 Baseline v2

- Timestamp: 2026-02-26 20:06:06
- Samples: 2801 | Features: 2000 (top-k variance)
- Labels: 91 raw classes → grouped with min_support=20 (OTHER=498)

## Model comparison (val)

| model | balanced_acc | macro_f1 |
|---|---:|---:|
| LinearSVC | 0.9546 | 0.9507 |
| LogReg(saga) | 0.9751 | 0.9688 |

Best (by macro_f1): **LogReg(saga)**

## Top confusions (normalized, true → pred)

| true | pred | rate |
|---|---|---:|
| LGG, PA/GG ST | CONTR, REACT | 0.333 |
| GBM, MES | GBM, RTK II | 0.286 |
| GBM, RTK I | GBM, RTK II | 0.188 |
| LGG, DNT | LGG, PA/GG ST | 0.091 |
| A IDH, HG | A IDH | 0.083 |
| MB, SHH INF | MB, SHH CHL AD | 0.077 |
| O IDH | A IDH | 0.050 |
| MB, SHH CHL AD | MB, SHH INF | 0.048 |
| MB, G4 | MB, G3 | 0.029 |
| GBM, RTK II | GBM, MES | 0.028 |
| OTHER | A IDH, HG | 0.008 |
| OTHER | CONTR, REACT | 0.008 |
| OTHER | DMG, K27 | 0.008 |
| OTHER | GBM, RTK I | 0.008 |
| OTHER | LGG, GG | 0.008 |

## Worst per-class recall (val)

| class | recall |
|---|---:|
| LGG, PA/GG ST | 0.667 |
| GBM, MES | 0.714 |
| GBM, RTK I | 0.812 |
| LGG, DNT | 0.909 |
| A IDH, HG | 0.917 |
| MB, SHH INF | 0.923 |
| OTHER | 0.944 |
| O IDH | 0.950 |
| MB, SHH CHL AD | 0.952 |
| MB, G4 | 0.971 |
| GBM, RTK II | 0.972 |
| A IDH | 1.000 |
| ANA PA | 1.000 |
| ATRT, MYC | 1.000 |
| ATRT, SHH | 1.000 |

## Artifacts

- Raw CM: `D:/Study Material/LifeReset/Epigenetics/TumorTyping/results/latest/confusion_matrix_raw.png`
- Normalized CM: `D:/Study Material/LifeReset/Epigenetics/TumorTyping/results/latest/confusion_matrix_norm.png`
