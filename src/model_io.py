from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.config import MODEL_PATH


def save_model(model: Any, path: str | Path = MODEL_PATH) -> Path:
    """Persist a fitted model and return the resulting artifact path."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path = MODEL_PATH) -> Any:
    """Load a model artifact and validate the inference interface."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact was not found at '{model_path}'. Run: python -m src.train"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact '{model_path}'") from exc

    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(f"Model artifact '{model_path}' does not support probability prediction")
    return model
