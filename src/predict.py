from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from src.config import MODELS_DIR


EXAMPLE_CLIENT = {
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


def main():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model is not trained. Run: python -m src.train")

    payload = EXAMPLE_CLIENT
    if len(sys.argv) > 1:
        payload = json.loads(sys.argv[1])

    model = joblib.load(model_path)
    X = pd.DataFrame([payload])
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)

    print(json.dumps({"prediction": prediction, "probability_yes": probability}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
