from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DROP_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_data(path: str | None = None) -> pd.DataFrame:
    from src.config import RAW_DATA_PATH

    data_path = path or RAW_DATA_PATH
    return pd.read_csv(data_path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    # Normalize text columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip().str.lower()

    # The UCI dataset has no missing values, but this keeps ETL robust.
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].replace({"nan": "unknown", "": "unknown"}).fillna("unknown")
        else:
            df[column] = df[column].fillna(df[column].median())

    return df


def prepare_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    df = clean_data(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not found")

    y = df[target_column].map({"no": 0, "yes": 1})
    if y.isnull().any():
        raise ValueError("Target column must contain only yes/no values")

    drop_columns = [target_column] + [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_columns)

    return X, y.astype(int)


def split_dataset(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_feature_groups(X: pd.DataFrame):
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    return numeric_features, categorical_features
