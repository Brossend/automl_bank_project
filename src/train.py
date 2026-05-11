from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    N_TRIALS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TABLES_DIR,
    VALIDATION_SIZE,
)
from src.download_data import ensure_dataset
from src.etl import get_feature_groups, load_data, prepare_features, split_dataset

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to create report figures") from exc


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups(X)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_model(trial):
    model_name = trial.suggest_categorical(
        "model",
        ["logistic_regression", "random_forest", "extra_trees"],
    )

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("lr_C", 0.01, 10.0, log=True),
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 80, 250),
            max_depth=trial.suggest_int("rf_max_depth", 4, 18),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 8),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return ExtraTreesClassifier(
        n_estimators=trial.suggest_int("et_n_estimators", 80, 250),
        max_depth=trial.suggest_int("et_max_depth", 4, 18),
        min_samples_split=trial.suggest_int("et_min_samples_split", 2, 12),
        min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 8),
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def create_figures(y_test, y_pred, y_proba, metrics: dict[str, float]) -> None:
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    distribution = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["no", "yes"], distribution.values)
    plt.title("Target distribution")
    plt.xlabel("Subscribed term deposit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def main(n_trials: int = N_TRIALS) -> dict[str, float]:
    ensure_dataset()

    for directory in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = prepare_features(df)
    save_target_distribution(y)

    X_train_full, X_test, y_train_full, y_test = split_dataset(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor(X_train)

    import optuna

    def objective(trial) -> float:
        model = build_model(trial)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        valid_pred = pipeline.predict(X_valid)
        return f1_score(y_valid, valid_pred, zero_division=0)

    study = optuna.create_study(direction="maximize", study_name="bank_marketing_automl")
    study.optimize(objective, n_trials=n_trials)

    best_model = build_model(study.best_trial)
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train_full)),
            ("model", best_model),
        ]
    )
    final_pipeline.fit(X_train_full, y_train_full)

    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    joblib.dump(final_pipeline, MODELS_DIR / "best_model.joblib")
    pd.DataFrame(study.trials_dataframe()).to_csv(TABLES_DIR / "optuna_trials.csv", index=False)

    report = {
        "best_params": study.best_params,
        "best_validation_f1": float(study.best_value),
        "test_metrics": metrics,
        "rows": int(len(df)),
        "features_used": list(X.columns),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    create_figures(y_test, y_pred, y_proba, metrics)

    try:
        import mlflow
    except Exception:
        mlflow = None

    if mlflow is not None:
        mlflow.set_experiment("bank_marketing_automl")
        with mlflow.start_run(run_name="best_automl_pipeline"):
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(MODELS_DIR / "best_model.joblib"))
            for figure in FIGURES_DIR.glob("*.png"):
                mlflow.log_artifact(str(figure), artifact_path="figures")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
