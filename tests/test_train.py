import pandas as pd

from src.train import build_preprocessor, calculate_metrics


def test_build_preprocessor_can_transform_mixed_features():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "balance": [100, 200, 300],
            "job": ["admin", "student", "admin"],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3


def test_calculate_metrics_returns_expected_keys():
    metrics = calculate_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_proba=[0.1, 0.8, 0.4, 0.2],
    )

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0 <= metrics["f1"] <= 1
