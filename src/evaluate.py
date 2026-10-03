from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import EVALUATION_REPORT_PATH, MODEL_PATH
from src.download_data import ensure_dataset
from src.etl import load_data, prepare_features, split_dataset
from src.logging_utils import configure_logging
from src.model_io import load_model
from src.reporting import calculate_metrics, create_figures

logger = logging.getLogger(__name__)


def main(model_path: str | Path = MODEL_PATH) -> dict[str, float]:
    """Evaluate a persisted model on the deterministic held-out test set."""
    ensure_dataset()

    df = load_data()
    X, y = prepare_features(df)
    _, X_test, _, y_test = split_dataset(X, y)

    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    create_figures(y_test, y_pred, y_proba, metrics)

    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    configure_logging()
    main()
