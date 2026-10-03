from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    MODELS_DIR,
    N_TRIALS,
    OPTUNA_STUDY_NAME,
    OPTUNA_TRIALS_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TABLES_DIR,
    TRAINING_REPORT_PATH,
    VALIDATION_SIZE,
)
from src.download_data import ensure_dataset
from src.etl import get_feature_groups, load_data, prepare_features, split_dataset
from src.logging_utils import configure_logging
from src.model_io import save_model
from src.reporting import calculate_metrics, create_figures, save_target_distribution

if TYPE_CHECKING:
    import optuna
    from optuna.trial import FrozenTrial


logger = logging.getLogger(__name__)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for the feature types present in the training data."""
    numeric_features, categorical_features = get_feature_groups(X)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_model(trial: optuna.Trial | FrozenTrial) -> ClassifierMixin:
    """Build one of the existing classifiers from an Optuna trial."""
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

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 80, 250),
            max_depth=trial.suggest_int("et_max_depth", 4, 18),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 8),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unsupported model type selected by Optuna: {model_name}")


def build_pipeline(X: pd.DataFrame, model: ClassifierMixin) -> Pipeline:
    """Combine preprocessing and classification into one serializable pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def run_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int,
) -> optuna.Study:
    """Run deterministic Optuna optimization for the existing search space."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    import optuna

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_pipeline(X_train, build_model(trial))
        pipeline.fit(X_train, y_train)
        valid_pred = pipeline.predict(X_valid)
        return float(f1_score(y_valid, valid_pred, zero_division=0))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_STUDY_NAME,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def _log_to_mlflow(metrics: dict[str, float], best_params: dict[str, object]) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is unavailable; experiment artifacts were not logged")
        return

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="best_automl_pipeline"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        for figure in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(figure), artifact_path="figures")


def _create_output_directories() -> None:
    for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def main(n_trials: int = N_TRIALS) -> dict[str, float]:
    """Train, evaluate, persist, and log the best model pipeline."""
    ensure_dataset()
    _create_output_directories()

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

    study = run_hyperparameter_search(
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_trials=n_trials,
    )

    final_pipeline = build_pipeline(X_train_full, build_model(study.best_trial))
    final_pipeline.fit(X_train_full, y_train_full)

    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    save_model(final_pipeline, MODEL_PATH)
    study.trials_dataframe().to_csv(OPTUNA_TRIALS_PATH, index=False)

    report = {
        "best_params": study.best_params,
        "best_validation_f1": float(study.best_value),
        "test_metrics": metrics,
        "rows": int(len(df)),
        "features_used": list(X.columns),
    }
    TRAINING_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_figures(y_test, y_pred, y_proba, metrics)
    _log_to_mlflow(metrics, study.best_params)

    logger.info("Training completed; model saved to %s", MODEL_PATH)
    logger.info("Test metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    configure_logging()
    main()
