from __future__ import annotations

import json

import joblib

from src.config import MODELS_DIR, REPORTS_DIR
from src.download_data import ensure_dataset
from src.etl import load_data, prepare_features, split_dataset
from src.train import calculate_metrics, create_figures


def main():
    ensure_dataset()
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model is not trained. Run: python -m src.train")

    df = load_data()
    X, y = prepare_features(df)
    _, X_test, _, y_test = split_dataset(X, y)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    create_figures(y_test, y_pred, y_proba, metrics)

    output_path = REPORTS_DIR / "evaluation_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
