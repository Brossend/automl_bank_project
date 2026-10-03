from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODEL_PATH, PREDICTION_THRESHOLD
from src.etl import clean_data
from src.logging_utils import configure_logging
from src.model_io import load_model

logger = logging.getLogger(__name__)


EXAMPLE_CLIENT: dict[str, object] = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 12,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def predict_client(
    payload: Mapping[str, object],
    model_path: str | Path = MODEL_PATH,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict[str, int | float]:
    """Predict a single client response after validating its feature schema."""
    if not payload:
        raise ValueError("Prediction payload must be a non-empty JSON object")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Prediction threshold must be between 0 and 1")

    model = load_model(model_path)
    expected_features = [str(name) for name in getattr(model, "feature_names_in_", [])]
    missing_features = sorted(set(expected_features).difference(payload))
    if missing_features:
        raise ValueError(f"Prediction payload is missing features: {missing_features}")

    features = clean_data(pd.DataFrame([dict(payload)]))
    try:
        probability = float(model.predict_proba(features)[0, 1])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prediction payload contains incompatible feature values") from exc

    prediction = int(probability >= threshold)
    return {"prediction": prediction, "probability_yes": probability}


def _parse_payload(arguments: Sequence[str]) -> Mapping[str, Any]:
    if not arguments:
        return EXAMPLE_CLIENT

    try:
        payload = json.loads(arguments[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Prediction payload must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Prediction payload must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, int | float]:
    """Run inference for a JSON CLI payload or the documented example client."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    result = predict_client(_parse_payload(arguments))
    logger.info("Prediction result: %s", json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    configure_logging()
    main()
