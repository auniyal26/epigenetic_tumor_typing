from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, asdict
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
from sklearn.linear_model import LogisticRegression
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

    # performance
    chunksize: int = 20000
    pass1_sample_subsample: int = 150
    pass1_max_probes: int = 80000
    top_k_by_variance: int = 2000

    # label policy
    min_class_support: int = 20
    other_label: str = "OTHER"

    # split
    test_size: float = 0.25
    seed: int = 42

    # models
    svm_C: float = 1.0
    svm_max_iter: int = 20000

    lr_C: float = 1.0
    lr_max_iter: int = 2000


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
    if subsample and subsample < len(sample_cols):
        rng = np.random.default_rng(seed)
        sample_cols_pass1 = list(rng.choice(sample_cols, size=subsample, replace=False))
    else:
        sample_cols_pass1 = sample_cols

    usecols = ["ID_REF"] + sample_cols_pass1
    heap: List[Tuple[float, str]] = []

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    seen = 0
    for chunk in tqdm(reader, desc=f"Pass1 variance (subsample={len(sample_cols_pass1)})", unit="chunk"):
        seen += len(chunk)
        probe_ids = chunk["ID_REF"].astype(str).tolist()
        arr = chunk.drop(columns=["ID_REF"]).to_numpy(dtype=np.float32, copy=False)
        v = np.nanvar(arr, axis=1)

        for pid, vv in zip(probe_ids, v):
            if np.isnan(vv):
                continue
            vv = float(vv)
            if len(heap) < k:
                heapq.heappush(heap, (vv, pid))
            else:
                if vv > heap[0][0]:
                    heapq.heapreplace(heap, (vv, pid))

        if max_probes and seen >= max_probes:
            break

    return [pid for _, pid in heap]


def pass2_load_topk(
    beta_path: str,
    sample_cols: List[str],
    top_probes: Set[str],
    chunksize: int,
) -> pd.DataFrame:
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

    for chunk in tqdm(reader, desc="Pass2 load top-k (all samples)", unit="chunk"):
        chunk = chunk[chunk["ID_REF"].isin(top_probes)]
        if chunk.empty:
            continue
        chunk = chunk.set_index("ID_REF")
        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        kept.append(chunk)

    if not kept:
        raise RuntimeError("Pass2 loaded zero probes.")
    return pd.concat(kept, axis=0)


def group_rare_classes(y: pd.Series, min_support: int, other: str) -> Tuple[pd.Series, Dict[str, int]]:
    counts = y.value_counts()
    keep = counts[counts >= min_support].index
    y2 = y.where(y.isin(keep), other)
    stats = {
        "n_classes_raw": int(counts.shape[0]),
        "n_classes_kept": int(len(keep)),
        "n_classes_after": int(y2.nunique()),
        "n_other": int((y2 == other).sum()),
    }
    return y2, stats


def save_cm(cm: np.ndarray, labels: List[str], outpath: Path, title: str, normalize: bool = False):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".2f" if normalize else "d", xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def top_confusions(cm: np.ndarray, labels: List[str], top_n: int = 15) -> List[Tuple[str, str, float]]:
    """
    For normalized CM: returns top off-diagonal confusions (true->pred with highest rate).
    """
    out = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            if i == j:
                continue
            out.append((true_lab, pred_lab, float(cm[i, j])))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:top_n]


def main():
    cfg = Config()
    outdir = ensure_outdir(cfg)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    steps = [
        "Load labels",
        "Read beta header (SAMPLE cols)",
        "Pass1 top-k variance",
        "Pass2 load top-k",
        "QC + ML matrix",
        "Group rare labels",
        "Split train/val",
        "Train Model A (SVM)",
        "Train Model B (LogReg)",
        "Eval + error analysis + save",
    ]

    with tqdm(total=len(steps), desc="Baseline v2 (GSE90496)", unit="step") as p:
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

        X = pass2_load_topk(cfg.beta_matrix_path, sample_cols, top_set, chunksize=cfg.chunksize)
        p.update(1)

        # QC: remove constant probes (rare after variance top-k but keep it)
        var = X.var(axis=1, skipna=True)
        X = X.loc[var > 1e-12]
        X_ml = X.T.values  # (samples, features)
        p.update(1)

        # label grouping
        y_grouped, label_stats = group_rare_classes(y, cfg.min_class_support, cfg.other_label)
        y_ml = y_grouped.values
        p.update(1)

        # split (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            X_ml, y_ml, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_ml
        )
        p.update(1)

        # Model A: Linear SVM
        model_a = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=cfg.svm_C, class_weight="balanced", dual=True, max_iter=cfg.svm_max_iter, random_state=cfg.seed)),
        ])
        model_a.fit(X_train, y_train)
        p.update(1)

        # Model B: Multinomial Logistic Regression
        model_b = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.lr_C,
                solver="saga",
                penalty="l2",
                max_iter=cfg.lr_max_iter,
                class_weight="balanced",
                n_jobs=None,
                random_state=cfg.seed,
            )),
        ])
        model_b.fit(X_train, y_train)
        p.update(1)

        # Evaluate both
        def eval_model(name: str, model) -> Dict:
            pred = model.predict(X_val)
            bal = float(balanced_accuracy_score(y_val, pred))
            mf1 = float(f1_score(y_val, pred, average="macro"))
            rep = classification_report(y_val, pred, output_dict=True, zero_division=0)
            per_recall = {k: float(v["recall"]) for k, v in rep.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")}
            return {"name": name, "balanced_acc": bal, "macro_f1": mf1, "pred": pred, "per_class_recall": per_recall}

        res_a = eval_model("LinearSVC", model_a)
        res_b = eval_model("LogReg(saga)", model_b)

        # Error analysis based on best macro_f1
        best = res_a if res_a["macro_f1"] >= res_b["macro_f1"] else res_b
        labels = sorted(list(set(y_val) | set(best["pred"])))

        cm_raw = confusion_matrix(y_val, best["pred"], labels=labels)
        cm_norm = confusion_matrix(y_val, best["pred"], labels=labels, normalize="true")

        cm_raw_path = outdir / "confusion_matrix_raw.png"
        cm_norm_path = outdir / "confusion_matrix_norm.png"
        save_cm(cm_raw, labels, cm_raw_path, title=f"{best['name']} — Confusion Matrix (val, raw)", normalize=False)
        save_cm(cm_norm, labels, cm_norm_path, title=f"{best['name']} — Confusion Matrix (val, normalized)", normalize=True)

        # Top confusions (normalized)
        confs = top_confusions(cm_norm, labels, top_n=15)

        # Worst per-class recall
        per_rec = best["per_class_recall"]
        worst = sorted(per_rec.items(), key=lambda x: x[1])[:15]

        # Results table (md)
        results_md = outdir / "results_table_v2.md"
        lines = []
        lines += ["# Results — GSE90496 Baseline v2", ""]
        lines += [f"- Timestamp: {ts}"]
        lines += [f"- Samples: {X_ml.shape[0]} | Features: {X_ml.shape[1]} (top-k variance)"]
        lines += [f"- Labels: 91 raw classes → grouped with min_support={cfg.min_class_support} (OTHER={label_stats['n_other']})"]
        lines += [""]
        lines += ["## Model comparison (val)", ""]
        lines += ["| model | balanced_acc | macro_f1 |", "|---|---:|---:|"]
        for r in [res_a, res_b]:
            lines += [f"| {r['name']} | {r['balanced_acc']:.4f} | {r['macro_f1']:.4f} |"]
        lines += ["", f"Best (by macro_f1): **{best['name']}**", ""]
        lines += ["## Top confusions (normalized, true → pred)", ""]
        lines += ["| true | pred | rate |", "|---|---|---:|"]
        for t, pr, rate in confs:
            lines += [f"| {t} | {pr} | {rate:.3f} |"]
        lines += ["", "## Worst per-class recall (val)", ""]
        lines += ["| class | recall |", "|---|---:|"]
        for cls, rec in worst:
            lines += [f"| {cls} | {rec:.3f} |"]
        lines += ["", "## Artifacts", ""]
        lines += [f"- Raw CM: `{cm_raw_path.as_posix()}`"]
        lines += [f"- Normalized CM: `{cm_norm_path.as_posix()}`"]
        results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 8–10 line log
        run_log = outdir / "run_log_v2.txt"
        log_lines = [
            f"[{ts}] Baseline v2 on GSE90496 (real beta values)",
            f"Pass1: subsample={cfg.pass1_sample_subsample}, max_probes={cfg.pass1_max_probes}, top_k={cfg.top_k_by_variance}, chunksize={cfg.chunksize}",
            f"Features used: {X_ml.shape[1]} probes",
            f"Label policy: min_support={cfg.min_class_support}, OTHER={label_stats['n_other']} samples, classes_after={label_stats['n_classes_after']}",
            f"Split: train/val = {1.0 - cfg.test_size:.2f}/{cfg.test_size:.2f} stratified (seed={cfg.seed})",
            f"Model A: LinearSVC bal_acc={res_a['balanced_acc']:.4f} macro_f1={res_a['macro_f1']:.4f}",
            f"Model B: LogReg(saga) bal_acc={res_b['balanced_acc']:.4f} macro_f1={res_b['macro_f1']:.4f}",
            f"Best: {best['name']} (by macro_f1)",
            f"Saved: results_table_v2.md + confusion_matrix_raw/norm.png + metrics_v2.json",
            f"Runtime: {time.time() - t0:.1f}s",
        ]
        run_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        metrics = {
            "timestamp": ts,
            "config": asdict(cfg),
            "label_stats": label_stats,
            "n_samples": int(X_ml.shape[0]),
            "n_features": int(X_ml.shape[1]),
            "model_results": {
                "LinearSVC": {"balanced_acc": res_a["balanced_acc"], "macro_f1": res_a["macro_f1"]},
                "LogReg_saga": {"balanced_acc": res_b["balanced_acc"], "macro_f1": res_b["macro_f1"]},
            },
            "best_model": best["name"],
            "top_confusions": confs,
            "worst_recall": worst,
            "artifacts": {
                "results_table": results_md.as_posix(),
                "cm_raw": cm_raw_path.as_posix(),
                "cm_norm": cm_norm_path.as_posix(),
                "run_log": run_log.as_posix(),
            },
            "runtime_sec": float(time.time() - t0),
        }
        (outdir / "metrics_v2.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        p.update(1)

    print("Done.")
    print("Wrote:", results_md)
    print("Wrote:", cm_norm_path)
    print("Wrote:", run_log)


if __name__ == "__main__":
    main()