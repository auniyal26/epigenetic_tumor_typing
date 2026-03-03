# Data Card — GSE90496 (Public CNS Tumor Methylation Reference Set)

**Dataset name:** GSE90496 — DNA methylation-based classification of CNS tumors (reference set)  
**Source:** NCBI GEO (Series: **GSE90496**)  
**Local file inspected:** `GSE90496_series_matrix.txt.gz` (Series Matrix; metadata + labels)

---

## Where the data came from
- Public dataset hosted on **NCBI GEO** under accession **GSE90496**.

---

## What files exist (and what each one is for)
From the GEO series/supplementary files list:

- **Series Matrix (`.txt.gz`)** ✅ *(you already have this)*  
  - Contains **sample metadata** and **labels** (e.g., methylation class per sample).
  - Good for *inspection, label extraction, counts, and field discovery*.

- **Beta matrix (`GSE90496_beta.txt.gz`)** 
  - Contains the actual **methylation beta-value matrix** (the ML input `X`).
  - Typical format: rows = CpG probes (`cg...`), columns = samples (`GSM...`), values in **[0, 1]**.

- **RAW (`GSE90496_RAW.tar`)**  
  - Raw array output (often IDAT/intensity-level files).
  - Needed later for deeper QC and potential batch/technical fields.

- **Class description (`GSE90496_methylationclassdescription.xlsx`)**  
  - Label dictionary / descriptions for methylation classes.
  - Useful for mapping classes to more human-readable groups.

---

## X (input) and y (labels)
### X — model input
- Expected input for ML: methylation **beta values** per CpG probe.
- Stored in `GSE90496_beta.txt.gz`.
- Values are typically in **[0, 1]** (0 ≈ unmethylated, 1 ≈ methylated).

### y — labels
- Label field present per sample: **methylation class**.
- In the series matrix, labels appear in sample characteristics fields (e.g., `methylation class: ...`).

---

## Matrix shape (samples × features)
- **Samples:** **2801**

### To be confirmed from Beta matrix
- **Features:** Illumina HumanMethylation450 probes (expected ~450k+ CpGs; exact row count confirmed once `GSE90496_beta.txt.gz` is available).

**Working ML shape (expected):** roughly **2801 × ~450,000** after converting to `(n_samples × n_features)`.

---

## Labels available (what can be modeled)
- **Primary label:** methylation class (fine-grained CNS tumor categories).
- **Unique classes (from metadata inspection):** **91**
- Examples seen in label strings include categories like GBM variants, DMG K27, medulloblastoma groups, meningioma, etc.

---

## Missingness / batch fields present
### based on preliminary view of the Series Matrix
- **Label missingness:** none observed for methylation class in the series metadata (all samples appear to carry a class label).
- **Batch fields:** series matrix contains protocol/processing fields, but **does not obviously expose** per-sample slide/array position (Sentrix) style identifiers.

### What needs the Beta matrix / RAW files (deferred)
- **Methylation missingness:** compute once `GSE90496_beta.txt.gz` is loaded (per probe/per sample).
- **Batch/technical identifiers:** often easier to extract from RAW/IDAT metadata.
