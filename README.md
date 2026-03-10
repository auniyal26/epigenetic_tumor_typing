# Epigenetics — Tumor Typing (Public Baselines)

This repo contains a practical, runnable epigenetic tumor-typing baseline built on the public CNS methylation reference cohort **GSE90496** (GEO). The focus is **implementation-first**: load → QC → baseline models → evaluation → error analysis. No overengineering.

## Projects

### TumorTyping/
Main project folder with scripts and shipped artifacts.
- Dataset: **GSE90496** (2801 samples, 91 methylation classes)
- Input: methylation **beta values** in **[0, 1]** (Illumina 450K-style probes)
- Labels: **methylation class** (CNS tumor entities/subtypes)

See: `TumorTyping/README.md`

## Dataset/QC Card (v1)

- Samples: **2801**
- Classes (raw): **91**
- Rare classes: **15** classes <10 samples, **41** classes <20 samples

**Label policy for baselines**
- Group classes with support < **20** into **OTHER**
- Resulting classes: **51** (kept=50, OTHER=498 samples)

**Split policy (baseline)**
- Train/val: **0.75/0.25**
- Stratified: **True**
- Seed: **42**

**Batch/confound metadata**
- Material: **FFPE=1878**, **Frozen=923** (potential confound)

Artifacts (latest):
- `TumorTyping/results/latest/class_counts_top30_v1.png`
- `TumorTyping/results/latest/qc_beta_hist_v1.png`
- `TumorTyping/results/latest/dataset_qc_card_v1.md`

## What’s shipped

- Baseline v1: streamed top-K variance feature selection + linear classifier
- Baseline v2: second model + grouped-label error analysis (normalized confusion matrix + top confusions)
- Baseline v3: compact error-analysis artifact generated from saved metrics (no retraining)

## License

This repository is released under the MIT License (see `LICENSE`).