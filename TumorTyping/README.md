> Repo overview: see `../README.md`

# TumorTyping — GSE90496 Methylation Baselines

Minimal, runnable tumor-typing baselines using GSE90496 methylation beta values.

## Setup
Install deps:
- numpy, pandas, scikit-learn, matplotlib, tqdm

Place data files into `Data/` (see `Data/README.md`).

## Run
Dataset profile:
```bash
python scripts/00_profile_dataset.py

Baseline v1:

python scripts/01_baseline_v1.py

Baseline v2 (LogReg + SVC, grouped labels):

python scripts/02_baseline_v2.py

Material confound check:

python scripts/03_material_confound.py

Outputs are written to results/latest/ (not tracked by git).