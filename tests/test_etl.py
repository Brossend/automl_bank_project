import pandas as pd
import pytest

from src.etl import clean_data, get_feature_groups, load_data, prepare_features, split_dataset


def test_clean_data_removes_duplicates_and_normalizes_text():
    df = pd.DataFrame(
        {
            "age": [30, 30, None],
            "job": [" Management ", " Management ", None],
            "duration": [100, 100, 200],
            "y": ["yes", "yes", "no"],
        }
    )

    cleaned = clean_data(df)

    assert len(cleaned) == 2
    assert cleaned["job"].iloc[0] == "management"
    assert cleaned.isnull().sum().sum() == 0


def test_prepare_features_maps_target_and_drops_duration():
    df = pd.DataFrame(
        {
            "age": [30, 45, 50, 21],
            "job": ["admin.", "blue-collar", "student", "services"],
            "duration": [100, 200, 150, 80],
            "y": ["yes", "no", "yes", "no"],
        }
    )

    X, y = prepare_features(df)

    assert "y" not in X.columns
    assert "duration" not in X.columns
    assert set(y.tolist()) == {0, 1}


def test_split_dataset_keeps_rows():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "job": ["a", "b"] * 5,
            "duration": [1, 2] * 5,
            "y": ["yes", "no"] * 5,
        }
    )
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_feature_groups_detect_numeric_and_categorical():
    X = pd.DataFrame({"age": [20, 30], "job": ["admin", "student"]})
    numeric, categorical = get_feature_groups(X)

    assert numeric == ["age"]
    assert categorical == ["job"]


def test_load_data_reports_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset was not found"):
        load_data(missing_path)


def test_split_dataset_is_reproducible():
    df = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "job": ["a", "b"] * 10,
            "duration": [1, 2] * 10,
            "y": ["yes", "no"] * 10,
        }
    )
    X, y = prepare_features(df)

    first_split = split_dataset(X, y)
    second_split = split_dataset(X, y)

    for first, second in zip(first_split, second_split, strict=True):
        assert first.index.tolist() == second.index.tolist()
