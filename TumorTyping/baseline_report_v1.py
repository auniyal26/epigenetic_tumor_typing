from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BASE = Path(__file__).resolve().parent
LATEST = BASE / "results" / "latest"

METRICS = LATEST / "metrics_v2.json"
CM_NORM = LATEST / "confusion_matrix_norm.png"

OUT_MD = LATEST / "baseline_report_v1.md"
OUT_PNG = LATEST / "baseline_report_v1.png"


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not METRICS.exists():
        raise FileNotFoundError(f"Missing {METRICS}. Run baseline_v2 first.")
    if not CM_NORM.exists():
        raise FileNotFoundError(f"Missing {CM_NORM}. Run baseline_v2 first (normalized CM).")

    data = json.loads(METRICS.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    best = data.get("best_model", "UNKNOWN")
    model_results = data.get("model_results", {})
    top_conf = data.get("top_confusions", [])[:8]  # short

    # Build markdown report
    lines = []
    lines += ["# Baseline report v1 — GSE90496", ""]
    lines += [f"- Timestamp: {ts}"]
    lines += [f"- Best model: **{best}** (from v2)"]
    lines += [f"- Feature selection: streamed top-k variance (top_k={cfg.get('top_k_by_variance')}, max_probes={cfg.get('pass1_max_probes')})"]
    lines += [f"- Label policy: min_support={cfg.get('min_class_support')} → OTHER grouping"]
    lines += [""]

    lines += ["## Metrics (val)", ""]
    lines += ["| model | balanced_acc | macro_f1 |", "|---|---:|---:|"]
    for name, r in model_results.items():
        # stored as LinearSVC / LogReg_saga keys
        lines += [f"| {name} | {r['balanced_acc']:.4f} | {r['macro_f1']:.4f} |"]
    lines += [""]

    lines += ["## Confusion matrix", ""]
    lines += [f"- Normalized CM image: `{CM_NORM.as_posix()}`", ""]

    lines += ["## Top confusions (normalized true → pred)", ""]
    lines += ["| true | pred | rate |", "|---|---|---:|"]
    for t, p, rate in top_conf:
        lines += [f"| {t} | {p} | {rate:.3f} |"]
    lines += [""]

    lines += ["## Artifacts", ""]
    lines += [f"- Report image: `{OUT_PNG.as_posix()}`"]
    lines += [f"- Report markdown: `{OUT_MD.as_posix()}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Create 1 image artifact: normalized CM + top confusions text
    img = mpimg.imread(str(CM_NORM))

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Normalized Confusion Matrix (val)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_title("Top confusions (normalized)")

    text_lines = []
    for i, (t, p, rate) in enumerate(top_conf, start=1):
        text_lines.append(f"{i}. {t} → {p}: {rate:.3f}")
    ax2.text(0.0, 1.0, "\n".join(text_lines), va="top", fontsize=11)

    fig.suptitle(f"GSE90496 — Baseline report v1 | best={best}", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print("Wrote:", OUT_MD)
    print("Wrote:", OUT_PNG)


if __name__ == "__main__":
    main()