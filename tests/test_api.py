"""HTTP contract tests: status codes, validation and response schema."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import get_settings
from src.inference import InferenceService
from src.prediction_log import PredictionLog
from tests.conftest import CLASS_NAMES, image_bytes


@pytest.fixture()
def prediction_log(tmp_path: Path) -> PredictionLog:
    return PredictionLog(tmp_path)  # tests never write monitoring logs into the repository


@pytest.fixture()
def client(service: InferenceService, prediction_log: PredictionLog) -> Iterator[TestClient]:
    with TestClient(create_app(service=service, prediction_log=prediction_log)) as test_client:  # `with` runs startup
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "model-v1.0.0",
                               "classes": CLASS_NAMES}


def test_predict_success(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("bottle.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert body["label"] == body["top_k"][0]["label"]
    assert len(body["top_k"]) == 3
    assert body["inference_ms"] > 0
    assert len(body["request_id"]) == 32


def test_predict_writes_one_monitoring_record(client: TestClient, prediction_log: PredictionLog) -> None:
    response = client.post("/predict", files={"file": ("bottle.png", image_bytes(size=(80, 40)), "image/png")})
    lines = prediction_log.predictions_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == response.json()["request_id"]
    assert record["label"] == response.json()["label"]
    assert record["model_version"] == "model-v1.0.0"
    assert (record["width"], record["height"], record["aspect_ratio"]) == (80, 40, 2.0)
    assert {"timestamp", "confidence", "inference_ms", "brightness", "contrast", "saturation"} <= record.keys()


def test_rejected_requests_are_not_logged_as_predictions(client: TestClient, prediction_log: PredictionLog) -> None:
    client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert not prediction_log.predictions_path.exists()


def test_predict_rejects_non_image_content_type(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_corrupt_image(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("fake.png", b"not really a png", "image/png")})
    assert response.status_code == 400


def test_predict_requires_file(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_predict_rejects_oversized_upload(service: InferenceService, prediction_log: PredictionLog,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MAX_IMAGE_BYTES", "100")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(service=service, prediction_log=prediction_log)) as small_limit_client:
            response = small_limit_client.post("/predict", files={"file": ("big.png", image_bytes(size=(256, 256)), "image/png")})
        assert response.status_code == 413
    finally:
        get_settings.cache_clear()
