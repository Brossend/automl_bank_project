from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from src.config import DROP_COLUMNS, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the semicolon-delimited source dataset from a validated path."""
    data_path = Path(path) if path is not None else RAW_DATA_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset was not found at '{data_path}'")

    try:
        dataframe = pd.read_csv(data_path, sep=";")
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed to read dataset from '{data_path}'") from exc

    if dataframe.empty:
        raise ValueError(f"Dataset at '{data_path}' is empty")
    return dataframe


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, normalize strings, and fill expected missing values."""
    if df.empty:
        raise ValueError("Input dataframe must not be empty")

    df = df.copy()
    df = df.drop_duplicates()

    # The UCI dataset has no missing values, but this keeps ETL robust.
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = (
                df[column]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "unknown", "": "unknown"})
            )

    columns_with_missing_values = df.columns[df.isnull().any()].tolist()
    if columns_with_missing_values:
        raise ValueError(
            f"Cannot impute columns containing only missing values: {columns_with_missing_values}"
        )

    return df


def prepare_features(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and separate model features from the binary target."""
    df = clean_data(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not found")

    y = df[target_column].map({"no": 0, "yes": 1})
    if y.isnull().any():
        raise ValueError("Target column must contain only yes/no values")

    drop_columns = [target_column] + [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_columns)

    return X, y.astype(int)


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a deterministic stratified train/test split."""
    if X.empty or y.empty:
        raise ValueError("Features and target must not be empty")
    if len(X) != len(y):
        raise ValueError("Features and target must contain the same number of rows")
    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes for stratified splitting")

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names for preprocessing."""
    if X.empty:
        raise ValueError("Feature dataframe must not be empty")
    numeric_features = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features
