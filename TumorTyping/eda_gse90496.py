from __future__ import annotations

import gzip
from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "Data"

SERIES = DATA / "GSE90496_series_matrix.txt.gz"
BETA   = DATA / "GSE90496_beta.txt.gz"
DESC   = DATA / "GSE90496_methylationclassdescription.xlsx"


def head_gz_text(path: Path, n_lines: int = 8) -> list[str]:
    lines = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for _ in range(n_lines):
            lines.append(f.readline().rstrip("\n"))
    return lines


def series_quick(series_path: Path):
    gsm = None
    titles = None
    meth = None

    with pd.io.common.get_handle(str(series_path), "r", compression="infer") as h:
        for line in h.handle:
            if line.startswith("!Sample_geo_accession"):
                gsm = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
                if vals and vals[0].lower().startswith("methylation class:"):
                    meth = [v.split(":", 1)[1].strip() for v in vals]
            if gsm is not None and titles is not None and meth is not None:
                break

    print("\n[SERIES MATRIX]")
    print("GSM count:", len(gsm) if gsm else None)
    print("Title example:", titles[:3] if titles else None)
    print("Methylation class example:", meth[:3] if meth else None)


def beta_quick(beta_path: Path):
    print("\n[BETA FILE: HEADER + SHAPE PROBE]")
    # header only
    hdr = pd.read_csv(beta_path, sep="\t", compression="infer", nrows=0)
    cols = hdr.columns.tolist()
    print("n_columns:", len(cols))
    print("first 12 columns:", cols[:12])

    # read tiny sample of rows to infer structure
    df = pd.read_csv(beta_path, sep="\t", compression="infer", nrows=5, low_memory=False)
    print("\nfirst 5 rows (first 8 cols):")
    print(df.iloc[:, :8])

    # detect probe id column name
    probe_col = df.columns[0]
    print("\nprobe_id_col:", probe_col)
    print("probe_id examples:", df[probe_col].astype(str).tolist()[:5])

    # detect alternating beta/pval pattern
    det_cols = [c for c in cols if str(c).lower().startswith("detection pval")]
    sample_cols = [c for c in cols if str(c).lower().startswith("sample")]
    print("\npattern counts:")
    print("SAMPLE* columns:", len(sample_cols))
    print("Detection Pval* columns:", len(det_cols))
    if len(sample_cols) and len(det_cols):
        print("Looks like alternating (beta, pval) pairs.")
        # check numeric range for SAMPLE cols in the 5-row sample
        sample_preview = df[sample_cols[:5]].apply(pd.to_numeric, errors="coerce")
        print("SAMPLE preview min/max:", float(np.nanmin(sample_preview.values)), float(np.nanmax(sample_preview.values)))

        pval_preview = df[det_cols[:5]].apply(pd.to_numeric, errors="coerce")
        print("PVAL preview min/max:", float(np.nanmin(pval_preview.values)), float(np.nanmax(pval_preview.values)))


def xlsx_quick(desc_path: Path):
    print("\n[XLSX: CLASS DESCRIPTION]")
    if not desc_path.exists():
        print("No xlsx found:", desc_path)
        return

    xl = pd.ExcelFile(desc_path)
    print("sheets:", xl.sheet_names)
    for sh in xl.sheet_names[:2]:
        d = xl.parse(sh, nrows=8)
        print(f"\nSheet '{sh}' head (first 8 rows):")
        print(d.head(8))


def main():
    print("Paths:")
    print(" SERIES:", SERIES, "exists:", SERIES.exists())
    print(" BETA:  ", BETA, "exists:", BETA.exists())
    print(" DESC:  ", DESC, "exists:", DESC.exists())

    # Quick raw text peek (sometimes contains hints)
    print("\n[RAW TEXT PEEK: series matrix first lines]")
    for line in head_gz_text(SERIES, n_lines=6):
        print(line)

    series_quick(SERIES)
    beta_quick(BETA)
    xlsx_quick(DESC)


if __name__ == "__main__":
    main()
