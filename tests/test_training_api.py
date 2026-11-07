import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from training.train import generate, predict_json, preprocess, train, write_json


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate()
    write_json("artifacts/validation.json", {"dataset.md5": "test-fixture"})
    preprocess()
    return train()


def test_portable_model_predicts_and_probabilities_sum_to_one(trained):
    labels, probabilities = predict_json(trained, [[5.1, 3.5, 1.4, 0.2]])
    assert labels.tolist() == [0]
    assert abs(probabilities.sum() - 1) < 1e-8


def test_prediction_api_and_model_lineage(trained, monkeypatch):
    from api.main import app

    trained["run_id"] = "a" * 32
    Path("model.json").write_text(json.dumps(trained), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", "model.json")
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        result = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4, 0.2]]})
        assert result.json()["predictions"] == [0]
        assert client.get("/version").json()["run_id"] == "a" * 32
        assert client.post("/predict", json={"instances": [[1, 2]]}).status_code == 422


def test_missing_model_is_live_but_never_ready(tmp_path, monkeypatch):
    from api.main import app

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.json"))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"instances": [[1, 2, 3, 4]]}).status_code == 503


def test_failed_gate_cannot_create_approved_model(tmp_path, monkeypatch):
    from training.train import gate

    write_json(
        tmp_path / "evaluation.json",
        {"run_id": "a" * 32, "metrics": {"f1": 0.4}, "threshold": 0.85},
    )
    client = type("Client", (), {"set_tag": lambda *args: None})()
    monkeypatch.setattr("mlflow.MlflowClient", lambda: client)
    with pytest.raises(ValueError, match="Quality gate failed"):
        gate(str(tmp_path))
    assert not (tmp_path / "approved" / "model.json").exists()
