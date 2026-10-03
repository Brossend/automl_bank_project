from __future__ import annotations

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency failure
    raise RuntimeError("matplotlib is required to create report figures") from exc


def calculate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike,
) -> dict[str, float]:
    """Calculate the project's unchanged classification metric set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def create_figures(
    y_test: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike,
    metrics: dict[str, float],
) -> None:
    """Create evaluation figures in the configured reports directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[name] for name in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Model metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics.png", dpi=150)
    plt.close()


def save_target_distribution(y: pd.Series) -> None:
    """Create the target-distribution figure used by the training report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    distribution = y.value_counts().sort_index()
    counts = [float(value) for value in distribution.to_list()]
    plt.figure(figsize=(5, 4))
    plt.bar(["no", "yes"], counts)
    plt.title("Target distribution")
    plt.xlabel("Subscribed term deposit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()
