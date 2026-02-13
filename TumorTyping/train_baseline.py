from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_outdir() -> Path:
    outdir = project_root() / "results" / "latest"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_confusion_matrix(cm, class_names, outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_pca_explained_variance_plot(pca: PCA, outpath: Path):
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum)
    ax.set_title("PCA cumulative explained variance (train)")
    ax.set_xlabel("# components")
    ax.set_ylabel("cumulative explained variance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def simulate_methylation_beta_matrix_disjoint(
    n_samples: int = 600,
    n_features: int = 8000,
    n_classes: int = 4,
    n_inform_per_class: int = 250,
    shift: float = 0.30,
    seed: int = 42,
):
    """
    Better simulator:
    - Each class has its OWN informative probes (disjoint sets).
    - That makes this a true 4-class problem, not "two islands".

    Returns:
      X, y, class_names, inform_by_class (list of arrays)
    """
    rng = np.random.default_rng(seed)

    base = n_samples // n_classes
    sizes = [base] * n_classes
    sizes[0] += n_samples - sum(sizes)

    y = np.concatenate([np.full(s, k, dtype=int) for k, s in enumerate(sizes)])
    rng.shuffle(y)

    X = rng.beta(a=2.0, b=5.0, size=(n_samples, n_features)).astype(np.float32)

    total_inform = n_inform_per_class * n_classes
    if total_inform > n_features:
        raise ValueError("n_inform_per_class * n_classes must be <= n_features")

    perm = rng.permutation(n_features)
    inform_by_class = [perm[k*n_inform_per_class:(k+1)*n_inform_per_class] for k in range(n_classes)]

    for k in tqdm(range(n_classes), desc="Simulate: inject class-specific signal", unit="class", position=0):
        rows = np.where(y == k)[0]
        idx = inform_by_class[k]

        # Push this class's probes up; other probes remain noise.
        X[np.ix_(rows, idx)] = np.clip(X[np.ix_(rows, idx)] + shift, 0.0, 1.0)

    class_names = [f"Class_{k}" for k in range(n_classes)]
    return X, y, class_names, inform_by_class


def svm_feature_debug(svm: LinearSVC, inform_union: np.ndarray):
    """
    Debug whether informative probes get larger weights.
    Prints stats + overlaps across K.
    """
    coef = svm.coef_  # (n_classes, n_features)
    score = np.max(np.abs(coef), axis=0)

    inform_score = score[inform_union]
    non_mask = np.ones_like(score, dtype=bool)
    non_mask[inform_union] = False
    non_score = score[non_mask]

    print(
        "Weight stats | "
        f"inform: mean={inform_score.mean():.6e}, max={inform_score.max():.6e} | "
        f"non: mean={non_score.mean():.6e}, max={non_score.max():.6e}"
    )

    for K in [200, 500, 1000, 2000, 4000]:
        topK = np.argsort(score)[-K:]
        overlapK = len(set(topK).intersection(set(inform_union)))
        print(f"Overlap top-{K}: {overlapK}/{K} = {overlapK/K:.3f}")

    # return something machine-readable too
    top200 = np.argsort(score)[-200:]
    overlap200 = len(set(top200).intersection(set(inform_union)))
    return overlap200


def evaluate(y_true, y_pred):
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)
    return bal_acc, macro_f1, cm


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    outdir = ensure_outdir()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cfg = {
        "n_samples": 600,
        "n_features": 8000,
        "n_classes": 4,
        "n_inform_per_class": 250,  # total = 1000 informative probes
        "shift": 0.30,
        "seed": 42,
        "svm_C": 1.0,
        "pca_components": 200,
    }

    steps = [
        "Simulate data",
        "Split",
        "Baseline A: LinearSVC (no PCA)",
        "Baseline B: PCA -> LinearSVC",
        "Save plots + metrics",
    ]

    with tqdm(total=len(steps), desc="Baseline pipeline", unit="step", position=1) as p:
        # 1) Data
        X, y, class_names, inform_by_class = simulate_methylation_beta_matrix_disjoint(
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            n_classes=cfg["n_classes"],
            n_inform_per_class=cfg["n_inform_per_class"],
            shift=cfg["shift"],
            seed=cfg["seed"],
        )
        inform_union = np.unique(np.concatenate(inform_by_class))
        p.update(1)

        # 2) Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=cfg["seed"], stratify=y
        )
        p.update(1)

        # ----- Baseline A: No PCA -----
        clf_a = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("svm", LinearSVC(
                    C=cfg["svm_C"],
                    class_weight="balanced",
                    dual=True,        # n_features >> n_samples
                    max_iter=20000,
                    random_state=cfg["seed"],
                )),
            ]
        )
        clf_a.fit(X_train, y_train)
        y_pred_a = clf_a.predict(X_test)
        bal_a, f1_a, cm_a = evaluate(y_test, y_pred_a)

        print("\n=== Baseline A (no PCA) ===")
        print(f"Balanced accuracy: {bal_a:.4f} | Macro F1: {f1_a:.4f}")
        overlap200 = svm_feature_debug(clf_a.named_steps["svm"], inform_union)
        print(f"Top-200 overlap count: {overlap200}/200")
        p.update(1)

        # ----- Baseline B: PCA -> SVM -----
        clf_b = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(n_components=cfg["pca_components"], random_state=cfg["seed"])),
                ("svm", LinearSVC(
                    C=cfg["svm_C"],
                    class_weight="balanced",
                    dual=False,       # after PCA, features small
                    max_iter=20000,
                    random_state=cfg["seed"],
                )),
            ]
        )
        clf_b.fit(X_train, y_train)
        y_pred_b = clf_b.predict(X_test)
        bal_b, f1_b, cm_b = evaluate(y_test, y_pred_b)

        print("\n=== Baseline B (PCA -> SVM) ===")
        print(f"Balanced accuracy: {bal_b:.4f} | Macro F1: {f1_b:.4f}")
        p.update(1)

        # Save PCA explained variance plot
        pca_ev_path = outdir / "pca_explained_variance.png"
        save_pca_explained_variance_plot(clf_b.named_steps["pca"], pca_ev_path)

        # 4) Save plots + metrics
        cm_a_path = outdir / "confusion_matrix_no_pca.png"
        cm_b_path = outdir / "confusion_matrix_pca.png"

        save_confusion_matrix(cm_a, class_names, cm_a_path, title="Confusion Matrix (no PCA)")
        save_confusion_matrix(cm_b, class_names, cm_b_path, title="Confusion Matrix (PCA -> SVM)")

        metrics = {
            "timestamp": timestamp,
            "project_root": project_root().as_posix(),
            "outdir": outdir.as_posix(),
            "config": cfg,
            "baseline_no_pca": {"balanced_accuracy": bal_a, "macro_f1": f1_a},
            "baseline_pca": {"balanced_accuracy": bal_b, "macro_f1": f1_b},
            "informative_probes": {
                "total": int(len(inform_union)),
                "per_class": int(cfg["n_inform_per_class"]),
            },
            "plots": {
                "cm_no_pca": cm_a_path.as_posix(),
                "cm_pca": cm_b_path.as_posix(),
                "pca_explained_variance": pca_ev_path.as_posix(),
            },
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        log_lines = [
            f"[{timestamp}] Deep baseline debug run",
            f"Data: simulated beta matrix in [0,1], class-specific disjoint probe sets",
            f"Shape: X={X.shape}, classes={len(class_names)}",
            f"Split: train={X_train.shape[0]} test={X_test.shape[0]} (stratified)",
            f"Baseline A (no PCA): bal_acc={bal_a:.4f}, macro_f1={f1_a:.4f}",
            f"Baseline B (PCA->SVM, k={cfg['pca_components']}): bal_acc={bal_b:.4f}, macro_f1={f1_b:.4f}",
            f"Saved: {cm_a_path.as_posix()}",
            f"Saved: {cm_b_path.as_posix()}",
            f"Saved: {pca_ev_path.as_posix()}",
        ]
        (outdir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        p.update(1)

    print("\nConfig used:", cfg)
    print("Artifacts:", outdir.as_posix())


if __name__ == "__main__":
    main()
