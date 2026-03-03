from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    series_matrix_path: str = str(BASE_DIR / "Data" / "GSE90496_series_matrix.txt.gz")
    beta_matrix_path: str = str(BASE_DIR / "Data" / "GSE90496_beta.txt.gz")
    out_dir: str = str(BASE_DIR / "results" / "latest")

    # streamed selection perf
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

    # model
    lr_C: float = 1.0
    lr_max_iter: int = 2000

    # material parsing
    material_key: str = "material"   # expects strings like "Material: Frozen"
    allowed_materials: Tuple[str, str] = ("Frozen", "FFPE")  # canonical names


def ensure_outdir(cfg: Config) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------
# Series matrix parsing
# ----------------------------
def parse_series_labels_and_material(series_path: str, material_key: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - methylation_class (str)
      - material (str or None)
    aligned to sample order.
    """
    meth_class: Optional[List[str]] = None
    material_vals: Optional[List[str]] = None

    with pd.io.common.get_handle(series_path, "r", compression="infer") as h:
        for line in h.handle:
            if not line.startswith("!Sample_characteristics_ch1"):
                continue

            vals = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            if not vals:
                continue

            head = vals[0].split(":", 1)[0].strip().lower() if ":" in vals[0] else ""
            if head == "methylation class":
                meth_class = [v.split(":", 1)[1].strip() for v in vals]
            elif head == material_key.lower():
                material_vals = [v.split(":", 1)[1].strip() for v in vals]

            if meth_class is not None and material_vals is not None:
                break

    if meth_class is None:
        raise RuntimeError("Could not find 'methylation class:' in series matrix characteristics rows.")
    if material_vals is None:
        raise RuntimeError(f"Could not find '{material_key}:' in series matrix characteristics rows.")

    df = pd.DataFrame({
        "methylation_class": meth_class,
        "material_raw": material_vals,
    })
    return df


def canonicalize_material(s: str) -> str:
    s2 = str(s).strip().lower()
    if s2 == "frozen":
        return "Frozen"
    if s2 == "ffpe":
        return "FFPE"
    return str(s).strip()


# ----------------------------
# Beta matrix parsing (SAMPLE cols)
# ----------------------------
def get_sample_columns_only(beta_path: str) -> List[str]:
    hdr = pd.read_csv(beta_path, sep="\t", compression="infer", nrows=0)
    cols = hdr.columns.tolist()
    sample_cols = [c for c in cols if str(c).upper().startswith("SAMPLE ")]
    if not sample_cols:
        raise RuntimeError("No SAMPLE columns found in beta file header.")
    return sample_cols


def sample_index_from_col(col: str) -> int:
    # "SAMPLE 1" -> 0, "SAMPLE 2801" -> 2800
    toks = str(col).strip().split()
    if len(toks) != 2 or toks[0].upper() != "SAMPLE":
        raise ValueError(f"Unexpected SAMPLE column name: {col}")
    return int(toks[1]) - 1


# ----------------------------
# Streamed top-K by variance (subset-aware)
# ----------------------------
def pass1_topk_variance_subset(
    beta_path: str,
    sample_cols_subset: List[str],
    k: int,
    chunksize: int,
    subsample: int,
    max_probes: int,
    seed: int,
) -> List[str]:
    if subsample and subsample < len(sample_cols_subset):
        rng = np.random.default_rng(seed)
        cols_pass1 = list(rng.choice(sample_cols_subset, size=subsample, replace=False))
    else:
        cols_pass1 = sample_cols_subset

    usecols = ["ID_REF"] + cols_pass1
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
    for chunk in tqdm(reader, desc=f"Pass1 variance (subset n={len(cols_pass1)})", unit="chunk"):
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


def pass2_load_topk_subset(
    beta_path: str,
    sample_cols_subset: List[str],
    top_probes: Set[str],
    chunksize: int,
) -> pd.DataFrame:
    usecols = ["ID_REF"] + sample_cols_subset
    kept = []

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc=f"Pass2 load top-k (subset n={len(sample_cols_subset)})", unit="chunk"):
        chunk = chunk[chunk["ID_REF"].isin(top_probes)]
        if chunk.empty:
            continue
        chunk = chunk.set_index("ID_REF")
        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        kept.append(chunk)

    if not kept:
        raise RuntimeError("Pass2 loaded zero probes.")
    return pd.concat(kept, axis=0)


# ----------------------------
# Label grouping (same policy as v2)
# ----------------------------
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


# ----------------------------
# Plots
# ----------------------------
def plot_material_vs_top_labels(df: pd.DataFrame, outpath: Path, top_n: int = 20):
    """
    Stacked bar chart: top-N labels by support, split by material.
    df columns: methylation_class, material
    """
    counts = df["methylation_class"].value_counts().head(top_n)
    top_labels = counts.index.tolist()

    sub = df[df["methylation_class"].isin(top_labels)].copy()
    # pivot: label x material -> counts
    pivot = (
        sub.groupby(["methylation_class", "material"])
        .size()
        .unstack(fill_value=0)
        .loc[top_labels]
    )

    # normalize per label to show proportions
    prop = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(prop))
    for col in prop.columns:
        ax.bar(prop.index, prop[col].values, bottom=bottom, label=col)
        bottom += prop[col].values

    ax.set_title(f"Material split within top-{top_n} labels (proportions)")
    ax.set_ylabel("proportion")
    ax.set_xlabel("methylation class")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ----------------------------
# Training/eval
# ----------------------------
def train_eval_logreg(X: np.ndarray, y: np.ndarray, cfg: Config) -> Dict:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=cfg.lr_C,
            solver="saga",
            penalty="l2",
            max_iter=cfg.lr_max_iter,
            class_weight="balanced",
            n_jobs=None,
            random_state=cfg.seed,
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return {
        "balanced_acc": float(balanced_accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro")),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(pd.Series(y).nunique()),
    }


def main():
    cfg = Config()
    outdir = ensure_outdir(cfg)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    # 1) series: labels + material
    meta = parse_series_labels_and_material(cfg.series_matrix_path, cfg.material_key)
    meta["material"] = meta["material_raw"].map(canonicalize_material)

    # 2) beta header: sample columns
    sample_cols_all = get_sample_columns_only(cfg.beta_matrix_path)
    if len(sample_cols_all) != meta.shape[0]:
        raise RuntimeError(f"Mismatch: SAMPLE cols={len(sample_cols_all)} vs meta rows={meta.shape[0]}")

    # 3) material counts + md
    mat_counts = meta["material"].value_counts(dropna=False).to_dict()

    # Top label material skew plot
    plot_path = outdir / "material_vs_label_top20.png"
    plot_material_vs_top_labels(meta[["methylation_class", "material"]], plot_path, top_n=20)

    md_lines = []
    md_lines += ["# Material confound check — GSE90496", ""]
    md_lines += [f"- Timestamp: {ts}", ""]
    md_lines += ["## Material counts", ""]
    md_lines += ["| material | n |", "|---|---:|"]
    for k, v in sorted(mat_counts.items(), key=lambda x: (-x[1], x[0])):
        md_lines += [f"| {k} | {int(v)} |"]
    md_lines += ["", f"Plot saved: `{plot_path.as_posix()}`", ""]
    (outdir / "material_counts.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 4) subset indices
    idx_frozen = meta.index[meta["material"] == "Frozen"].to_numpy()
    idx_ffpe = meta.index[meta["material"] == "FFPE"].to_numpy()

    def cols_for_idx(idxs: np.ndarray) -> List[str]:
        # idxs are 0..2800 in sample order; match to "SAMPLE i+1"
        keep = set(int(i) for i in idxs.tolist())
        return [c for c in sample_cols_all if sample_index_from_col(c) in keep]

    cols_frozen = cols_for_idx(idx_frozen)
    cols_ffpe = cols_for_idx(idx_ffpe)

    # 5) run baseline per subset (selection+load within subset)
    results: Dict[str, Dict] = {}
    for name, idxs, cols in [
        ("Frozen", idx_frozen, cols_frozen),
        ("FFPE", idx_ffpe, cols_ffpe),
    ]:
        y = meta.loc[idxs, "methylation_class"].reset_index(drop=True)

        # label grouping (same as baseline v2)
        y_grouped, label_stats = group_rare_classes(y, cfg.min_class_support, cfg.other_label)

        # streamed top-k selection within subset
        top_list = pass1_topk_variance_subset(
            cfg.beta_matrix_path,
            cols,
            k=cfg.top_k_by_variance,
            chunksize=cfg.chunksize,
            subsample=cfg.pass1_sample_subsample,
            max_probes=cfg.pass1_max_probes,
            seed=cfg.seed,
        )
        top_set = set(top_list)

        X = pass2_load_topk_subset(
            cfg.beta_matrix_path,
            cols,
            top_set,
            chunksize=cfg.chunksize,
        )

        # QC: drop constant
        var = X.var(axis=1, skipna=True)
        X = X.loc[var > 1e-12]

        X_ml = X.T.values
        y_ml = y_grouped.values

        met = train_eval_logreg(X_ml, y_ml, cfg)
        results[name] = {
            "n_samples": int(X_ml.shape[0]),
            "n_features": int(X_ml.shape[1]),
            "label_stats": label_stats,
            "metrics": met,
        }

    payload = {
        "timestamp": ts,
        "config": asdict(cfg),
        "material_counts": mat_counts,
        "subset_results": results,
        "artifacts": {
            "material_counts_md": (outdir / "material_counts.md").as_posix(),
            "material_vs_label_top20_png": plot_path.as_posix(),
            "baseline_material_split_json": (outdir / "baseline_material_split.json").as_posix(),
            "run_log": (outdir / "run_log_material.txt").as_posix(),
        },
        "runtime_sec": float(time.time() - t0),
    }

    (outdir / "baseline_material_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 6-line run log
    lines = [
        f"[{ts}] Material confound check (GSE90496)",
        f"Material counts: {mat_counts}",
        f"Settings: topk={cfg.top_k_by_variance}, pass1_subsample={cfg.pass1_sample_subsample}, pass1_max_probes={cfg.pass1_max_probes}, min_support={cfg.min_class_support}",
        f"Frozen: bal_acc={results['Frozen']['metrics']['balanced_acc']:.4f}, macro_f1={results['Frozen']['metrics']['macro_f1']:.4f}, n={results['Frozen']['n_samples']}",
        f"FFPE:   bal_acc={results['FFPE']['metrics']['balanced_acc']:.4f}, macro_f1={results['FFPE']['metrics']['macro_f1']:.4f}, n={results['FFPE']['n_samples']}",
        "Interpretation: if macro_f1 differs by >0.03–0.05, Material is likely a real confound; otherwise lower risk.",
        f"Runtime: {payload['runtime_sec']:.1f}s",
    ]
    (outdir / "run_log_material.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Done.")
    print("Wrote:", (outdir / "material_counts.md"))
    print("Wrote:", plot_path)
    print("Wrote:", (outdir / "baseline_material_split.json"))
    print("Wrote:", (outdir / "run_log_material.txt"))


if __name__ == "__main__":
    main()