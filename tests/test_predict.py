import joblib
import numpy as np
import pytest

from src.predict import _parse_payload, predict_client


class StubProbabilityModel:
    feature_names_in_ = ["age", "job"]

    def predict_proba(self, features):
        assert features.loc[0, "job"] == "management"
        return np.array([[0.25, 0.75]])


def test_predict_client_validates_and_normalizes_payload(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(StubProbabilityModel(), model_path)

    result = predict_client(
        {"age": 35, "job": " Management "},
        model_path=model_path,
    )

    assert result == {"prediction": 1, "probability_yes": 0.75}


def test_predict_client_rejects_missing_features(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(StubProbabilityModel(), model_path)

    with pytest.raises(ValueError, match="missing features"):
        predict_client({"age": 35}, model_path=model_path)


def test_parse_payload_rejects_non_object_json():
    with pytest.raises(ValueError, match="JSON object"):
        _parse_payload(["[]"])
