from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
LATEST = BASE / "results" / "latest"
METRICS = LATEST / "metrics_v2.json"

OUT_PNG = LATEST / "artifact_v3_error_analysis.png"
OUT_LOG = LATEST / "artifact_v3_log.txt"


def main():
    if not METRICS.exists():
        raise FileNotFoundError(f"Missing: {METRICS}. Run baseline_v2 first to create metrics_v2.json")

    data = json.loads(METRICS.read_text(encoding="utf-8"))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best = data.get("best_model", "UNKNOWN")
    cfg = data.get("config", {})
    model_results = data.get("model_results", {})

    top_conf = data.get("top_confusions", [])   # list of [true, pred, rate]
    worst = data.get("worst_recall", [])        # list of [class, recall]

    # Defensive: keep only meaningful confusions (ignore OTHER->* if you want cleaner plot)
    # Here we keep as-is but cap at 12.
    top_conf = top_conf[:12]
    worst = worst[:12]

    # Prepare plot data
    conf_labels = [f"{t} → {p}" for (t, p, r) in top_conf]
    conf_rates = [r for (t, p, r) in top_conf]

    worst_labels = [c for (c, r) in worst]
    worst_recalls = [r for (c, r) in worst]

    # Plot: one image, two panels (readable + shippable)
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.barh(list(reversed(worst_labels)), list(reversed(worst_recalls)))
    ax1.set_title("Worst per-class recall (val)")
    ax1.set_xlabel("recall")
    ax1.set_xlim(0.0, 1.0)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.barh(list(reversed(conf_labels)), list(reversed(conf_rates)))
    ax2.set_title("Top confusions (normalized, true → pred)")
    ax2.set_xlabel("rate")
    ax2.set_xlim(0.0, 1.0)

    fig.suptitle(f"GSE90496 — Baseline v3 Artifact (from metrics_v2.json) | best={best}", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    # Short log (8–10 lines)
    lines = []
    lines.append(f"[{ts}] Artifact v3 shipped — error analysis summary")
    lines.append(f"Input: {METRICS.as_posix()}")
    lines.append(f"Best model (from v2): {best}")
    if "LinearSVC" in model_results and "LogReg_saga" in model_results:
        a = model_results["LinearSVC"]
        b = model_results["LogReg_saga"]
        lines.append(f"v2 scores: LinearSVC macro_f1={a['macro_f1']:.4f}, LogReg(saga) macro_f1={b['macro_f1']:.4f}")
    lines.append(f"Config snapshot: top_k={cfg.get('top_k_by_variance')} max_probes={cfg.get('pass1_max_probes')} min_support={cfg.get('min_class_support')}")
    lines.append(f"Top confusions plotted: {len(top_conf)}")
    lines.append(f"Worst recall classes plotted: {len(worst)}")
    lines.append(f"Saved artifact: {OUT_PNG.as_posix()}")
    lines.append("Note: this artifact is derived from v2 metrics (no re-training / no data reload).")

    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote:", OUT_PNG)
    print("Wrote:", OUT_LOG)


if __name__ == "__main__":
    main()