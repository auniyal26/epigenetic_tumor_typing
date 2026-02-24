from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "Data"
OUTDIR = BASE_DIR / "results" / "latest"
OUTDIR.mkdir(parents=True, exist_ok=True)

SERIES = DATA / "GSE90496_series_matrix.txt.gz"


def parse_series_rows(series_path: Path):
    """
    Parse key per-sample rows from the series matrix.
    Returns dict of lists aligned to sample order.
    """
    rows = {
        "gsm": None,
        "title": None,
        "methylation_class": None,
        "characteristics_rows": [],  # list of (row_name, values)
    }

    with pd.io.common.get_handle(str(series_path), "r", compression="infer") as h:
        for line in h.handle:
            if line.startswith("!Sample_geo_accession"):
                rows["gsm"] = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                rows["title"] = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
                # store the row;
                rows["characteristics_rows"].append(vals)
                # capture methylation class row
                if vals and vals[0].lower().startswith("methylation class:"):
                    rows["methylation_class"] = [v.split(":", 1)[1].strip() for v in vals]
            # stop early once we have core rows + a few characteristics rows
            # (series matrix has many rows; but this is still light)
            if rows["gsm"] and rows["title"] and rows["methylation_class"] and len(rows["characteristics_rows"]) >= 12:
                break

    if not (rows["gsm"] and rows["title"] and rows["methylation_class"]):
        raise RuntimeError("Could not find gsm/title/methylation class in series matrix.")

    return rows


def extract_sample_number(title: str) -> int | None:
    """
    Titles look like: 'GBM, G34, sample 1 [reference set]'
    Extracts the sample number for potential mapping/debugging.
    """
    m = re.search(r"\bsample\s+(\d+)\b", title.lower())
    return int(m.group(1)) if m else None


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = parse_series_rows(SERIES)

    df = pd.DataFrame({
        "gsm": rows["gsm"],
        "title": rows["title"],
        "methylation_class": rows["methylation_class"],
    })

    df["sample_number_in_title"] = df["title"].map(extract_sample_number)

    # ---- label distribution ----
    counts = df["methylation_class"].value_counts()
    n_samples = int(df.shape[0])
    n_classes = int(counts.shape[0])

    # Rare class stats
    rare_lt5 = int((counts < 5).sum())
    rare_lt10 = int((counts < 10).sum())
    rare_lt20 = int((counts < 20).sum())

    # Top 30 plot
    topN = 30
    top = counts.head(topN).iloc[::-1]  # reverse for horizontal bar
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top.index, top.values)
    ax.set_title(f"GSE90496 methylation class counts — top {topN}")
    ax.set_xlabel("samples")
    fig.tight_layout()
    plot_path = OUTDIR / "class_counts_top30.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    # ---- batch-ish fields discovery ----
    # characteristics rows are lists of strings like "key: value"
    char_rows = rows["characteristics_rows"]
    row_summaries = []
    for i, vals in enumerate(char_rows[:12]):
        # field name is whatever appears before ":" in the first sample
        first = vals[0]
        key = first.split(":", 1)[0].strip().lower() if ":" in first else "unknown"
        # count unique values across samples (rough proxy)
        cleaned = [v.split(":", 1)[1].strip() if ":" in v else v.strip() for v in vals]
        nunique = len(set(cleaned))
        example_vals = list(dict.fromkeys(cleaned))[:5]
        row_summaries.append({
            "row_idx": i,
            "key_guess": key,
            "n_unique": nunique,
            "examples": example_vals,
        })

    # ---- write data_profile_v2.md ----
    md = []
    md.append(f"# Data Profile v2 — GSE90496 (Labels + Batch Scan)\n")
    md.append(f"- Timestamp: {ts}")
    md.append(f"- Source file: `{SERIES.name}`")
    md.append("")
    md.append("## Labels: methylation_class")
    md.append(f"- Samples: **{n_samples}**")
    md.append(f"- Unique classes: **{n_classes}**")
    md.append(f"- Class imbalance:")
    md.append(f"  - classes with <5 samples: **{rare_lt5}**")
    md.append(f"  - classes with <10 samples: **{rare_lt10}**")
    md.append(f"  - classes with <20 samples: **{rare_lt20}**")
    md.append("")
    md.append("### Top 15 classes by sample count")
    md.append("")
    md.append("| class | n |")
    md.append("|---|---:|")
    for cls, n in counts.head(15).items():
        md.append(f"| {cls} | {int(n)} |")
    md.append("")
    md.append(f"Plot saved: `{plot_path.as_posix()}`")
    md.append("")
    md.append("## Batch/metadata scan (Series Matrix characteristics rows)")
    md.append("These are potential batch/confound fields if they exist (site, platform, tissue, processing, etc.).")
    md.append("We scan the first ~12 `!Sample_characteristics_ch1` rows and summarize unique values.")
    md.append("")
    md.append("| row_idx | key_guess | n_unique | examples |")
    md.append("|---:|---|---:|---|")
    for r in row_summaries:
        ex = "; ".join([str(x) for x in r["examples"]])
        md.append(f"| {r['row_idx']} | {r['key_guess']} | {r['n_unique']} | {ex} |")
    md.append("")
    md.append("## Suggested split policy (practical)")
    md.append("- Use stratified train/val split, but consider a minimum support threshold for classes (e.g., drop or group classes with <10 samples).")
    md.append("- Report macro metrics + per-class recall; for rare classes, metrics are unstable (high variance).")
    md.append("")
    md_path = OUTDIR / "data_profile_v2.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Also save a machine-readable JSON
    json_path = OUTDIR / "data_profile_v2.json"
    payload = {
        "timestamp": ts,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "rare_counts": {"lt5": rare_lt5, "lt10": rare_lt10, "lt20": rare_lt20},
        "top15": counts.head(15).to_dict(),
        "characteristics_scan": row_summaries,
        "artifacts": {"md": md_path.as_posix(), "plot": plot_path.as_posix(), "json": json_path.as_posix()},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Wrote:", md_path)
    print("Wrote:", plot_path)
    print("Wrote:", json_path)


if __name__ == "__main__":
    main()