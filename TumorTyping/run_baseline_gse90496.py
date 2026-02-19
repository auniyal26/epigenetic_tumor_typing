from __future__ import annotations

import heapq
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    series_matrix_path: str = str(BASE_DIR / "Data" / "GSE90496_series_matrix.txt.gz")
    beta_matrix_path: str = str(BASE_DIR / "Data" / "GSE90496_beta.txt.gz")
    out_dir: str = str(BASE_DIR / "results" / "latest")

    # -------- Speed controls (today: ship baseline fast) --------
    chunksize: int = 20000               # probes per chunk (drop to 10000 if RAM issues)
    pass1_sample_subsample: int = 100    # scan variance using only 100 samples
    pass1_max_probes: int = 80000        # scan only first 80k probes in pass1 (approx top-k)

    # Feature count
    top_k_by_variance: int = 2000        # start 2000 today; later 5000/10000

    # QC
    constant_tol: float = 1e-12

    # Split
    test_size: float = 0.25
    seed: int = 42

    # Model
    svm_C: float = 1.0
    max_iter: int = 20000


def ensure_outdir(cfg: Config) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_labels_in_order(series_path: str) -> pd.Series:
    meth_class = None
    with pd.io.common.get_handle(series_path, "r", compression="infer") as h:
        for line in h.handle:
            if line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
                if vals and vals[0].lower().startswith("methylation class:"):
                    meth_class = [v.split(":", 1)[1].strip() for v in vals]
                    break
    if meth_class is None:
        raise RuntimeError("Could not find methylation class row in series matrix.")
    return pd.Series(meth_class, name="methylation_class")


def get_sample_columns_only(beta_path: str) -> List[str]:
    """
    Beta columns are: ID_REF, SAMPLE 1, Detection Pval, SAMPLE 2, Detection Pval.1, ...
    We only want SAMPLE* columns.
    """
    hdr = pd.read_csv(beta_path, sep="\t", compression="infer", nrows=0)
    cols = hdr.columns.tolist()
    sample_cols = [c for c in cols if str(c).upper().startswith("SAMPLE ")]
    if not sample_cols:
        raise RuntimeError("No SAMPLE columns found in beta file header.")
    return sample_cols


def pass1_topk_variance(
    beta_path: str,
    sample_cols: List[str],
    k: int,
    chunksize: int,
    subsample: int,
    max_probes: int,
    seed: int,
) -> List[str]:
    """
    Pass1: stream through probes and keep top-k probe IDs by variance.

    Speed:
      - compute variance using only a random subsample of samples
      - stop after max_probes (approx top-k; enough for today's baseline)
    """
    # Subsample sample columns for fast variance ranking
    if subsample and subsample < len(sample_cols):
        rng = np.random.default_rng(seed)
        sample_cols_pass1 = list(rng.choice(sample_cols, size=subsample, replace=False))
    else:
        sample_cols_pass1 = sample_cols

    usecols = ["ID_REF"] + sample_cols_pass1
    heap: List[Tuple[float, str]] = []  # min-heap (var, probe)

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    seen = 0
    for chunk in tqdm(reader, desc=f"Pass1: variance scan (subsample={len(sample_cols_pass1)})", unit="chunk"):
        seen += len(chunk)
        probe_ids = chunk["ID_REF"].astype(str).tolist()

        vals_df = chunk.drop(columns=["ID_REF"])
        # fast numpy variance (still parsing is the slow bit, but this avoids pandas var)
        arr = vals_df.to_numpy(dtype=np.float32, copy=False)
        v = np.nanvar(arr, axis=1)

        for probe_id, vv in zip(probe_ids, v):
            if np.isnan(vv):
                continue
            vv = float(vv)
            if len(heap) < k:
                heapq.heappush(heap, (vv, probe_id))
            else:
                if vv > heap[0][0]:
                    heapq.heapreplace(heap, (vv, probe_id))

        if max_probes and seen >= max_probes:
            break

    return [pid for _, pid in heap]


def pass2_load_topk(
    beta_path: str,
    sample_cols: List[str],
    top_probes: Set[str],
    chunksize: int,
) -> pd.DataFrame:
    """
    Pass2: stream again and keep only selected probes; return probes × samples.
    Uses ALL samples for training/eval.
    """
    usecols = ["ID_REF"] + sample_cols
    kept = []

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Pass2: load top-k (all samples)", unit="chunk"):
        chunk = chunk[chunk["ID_REF"].isin(top_probes)]
        if chunk.empty:
            continue
        chunk = chunk.set_index("ID_REF")
        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        kept.append(chunk)

    if not kept:
        raise RuntimeError("Pass2 loaded zero probes. Probe IDs mismatch?")
    return pd.concat(kept, axis=0)


def save_confusion_matrix(cm: np.ndarray, labels: List[str], outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    cfg = Config()
    outdir = ensure_outdir(cfg)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    steps = [
        "Load labels (in order)",
        "Read beta header (SAMPLE cols)",
        "Pass1: top-k variance scan (chunked, subsampled, capped probes)",
        "Pass2: load top-k probes (chunked, all samples)",
        "QC (constant filter)",
        "Split train/val",
        "Train LinearSVC",
        "Evaluate + save",
    ]

    with tqdm(total=len(steps), desc="Baseline (GSE90496 ship-fast)", unit="step") as p:
        y = parse_labels_in_order(cfg.series_matrix_path)
        p.update(1)

        sample_cols = get_sample_columns_only(cfg.beta_matrix_path)
        if len(sample_cols) != len(y):
            raise RuntimeError(f"Mismatch: SAMPLE cols={len(sample_cols)} vs labels={len(y)}")
        p.update(1)

        top_list = pass1_topk_variance(
            cfg.beta_matrix_path,
            sample_cols,
            k=cfg.top_k_by_variance,
            chunksize=cfg.chunksize,
            subsample=cfg.pass1_sample_subsample,
            max_probes=cfg.pass1_max_probes,
            seed=cfg.seed,
        )
        top_set = set(top_list)
        p.update(1)

        X = pass2_load_topk(
            cfg.beta_matrix_path,
            sample_cols,
            top_set,
            chunksize=cfg.chunksize,
        )
        p.update(1)

        qc_stats: Dict[str, int] = {}
        qc_stats["probes_loaded_topk"] = int(X.shape[0])

        var = X.var(axis=1, skipna=True)
        X = X.loc[var > cfg.constant_tol]
        qc_stats["probes_after_constant_filter"] = int(X.shape[0])
        p.update(1)

        X_ml = X.T.values
        y_ml = y.values

        X_train, X_val, y_train, y_val = train_test_split(
            X_ml,
            y_ml,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y_ml,
        )
        p.update(1)

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("svm", LinearSVC(
                    C=cfg.svm_C,
                    class_weight="balanced",
                    dual=True,
                    max_iter=cfg.max_iter,
                    random_state=cfg.seed,
                )),
            ]
        )
        clf.fit(X_train, y_train)
        p.update(1)

        y_pred = clf.predict(X_val)
        bal_acc = float(balanced_accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))

        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        per_class_recall = {k: float(v["recall"]) for k, v in report.items()
                            if k not in ("accuracy", "macro avg", "weighted avg")}

        classes = sorted(list(set(y_val) | set(y_pred)))
        cm = confusion_matrix(y_val, y_pred, labels=classes)

        cm_path = outdir / "confusion_matrix.png"
        save_confusion_matrix(cm, classes, cm_path, title="GSE90496 — Confusion Matrix (val)")

        results_md = outdir / "results_table.md"
        results_md.write_text(
            "\n".join([
                "# Results — GSE90496 Baseline (ship-fast streamed top-k)",
                "",
                f"- Timestamp: {ts}",
                f"- Samples: {X_ml.shape[0]}",
                f"- Features used: {X_ml.shape[1]} (top-k variance; pass1 capped)",
                f"- Pass1 variance subsample: {cfg.pass1_sample_subsample} samples",
                f"- Pass1 max probes scanned: {cfg.pass1_max_probes}",
                f"- Split: train/val = {1.0 - cfg.test_size:.2f}/{cfg.test_size:.2f} (stratified, seed={cfg.seed})",
                f"- Model: StandardScaler + LinearSVC(C={cfg.svm_C}, class_weight=balanced)",
                "",
                "## Metrics",
                "| metric | value |",
                "|---|---:|",
                f"| balanced_accuracy (val) | {bal_acc:.4f} |",
                f"| macro_f1 (val) | {macro_f1:.4f} |",
                "",
                "## QC",
                f"- probes_loaded_topk: {qc_stats['probes_loaded_topk']}",
                f"- probes_after_constant_filter: {qc_stats['probes_after_constant_filter']}",
                "",
                "## Artifacts",
                f"- Confusion matrix: `{cm_path.as_posix()}`",
            ]) + "\n",
            encoding="utf-8"
        )

        run_log = outdir / "run_log.txt"
        log_lines = [
            f"[{ts}] GSE90496 baseline run (ship-fast)",
            f"Files: series={Path(cfg.series_matrix_path).name} beta={Path(cfg.beta_matrix_path).name}",
            f"Beta format: ID_REF + 2801 SAMPLE cols + 2801 Detection Pval cols (ignored)",
            f"Labels: methylation class (N={len(y)})",
            f"Pass1: subsample={cfg.pass1_sample_subsample}, max_probes={cfg.pass1_max_probes}, top_k={cfg.top_k_by_variance}, chunksize={cfg.chunksize}",
            f"Pass2: loaded top_k probes -> {qc_stats['probes_loaded_topk']} probes (all samples)",
            f"QC: constant_tol={cfg.constant_tol} -> {qc_stats['probes_after_constant_filter']} probes",
            f"Split: train/val = {1.0 - cfg.test_size:.2f}/{cfg.test_size:.2f}, stratified, seed={cfg.seed}",
            f"Model: StandardScaler + LinearSVC(C={cfg.svm_C}, balanced, max_iter={cfg.max_iter})",
            f"Metrics (val): bal_acc={bal_acc:.4f}, macro_f1={macro_f1:.4f}",
        ]
        run_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        metrics = {
            "timestamp": ts,
            "config": asdict(cfg),
            "X_shape_samples_by_features": [int(X_ml.shape[0]), int(X_ml.shape[1])],
            "qc_stats": qc_stats,
            "split": {"train_frac": float(1.0 - cfg.test_size), "val_frac": float(cfg.test_size), "seed": int(cfg.seed)},
            "model": {"type": "LinearSVC", "C": float(cfg.svm_C), "class_weight": "balanced", "max_iter": int(cfg.max_iter)},
            "metrics": {"balanced_accuracy_val": bal_acc, "macro_f1_val": macro_f1},
            "per_class_recall_val": per_class_recall,
            "artifacts": {
                "confusion_matrix": cm_path.as_posix(),
                "results_table": results_md.as_posix(),
                "run_log": run_log.as_posix(),
            },
            "runtime_sec": float(time.time() - t0),
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        p.update(1)

    print(f"Balanced accuracy (val): {bal_acc:.4f}")
    print(f"Macro F1 (val): {macro_f1:.4f}")
    print(f"Saved CM: {cm_path}")
    print(f"Artifacts: {outdir.as_posix()}")


if __name__ == "__main__":
    main()
