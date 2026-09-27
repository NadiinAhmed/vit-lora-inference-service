"""HTTP contract tests: status codes, validation and response schema."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import get_settings
from src.inference import InferenceService
from tests.conftest import CLASS_NAMES, image_bytes


@pytest.fixture()
def client(service: InferenceService) -> Iterator[TestClient]:
    with TestClient(create_app(service=service)) as test_client:  # `with` runs startup (lifespan)
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "classes": CLASS_NAMES}


def test_predict_success(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("bottle.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert body["label"] == body["top_k"][0]["label"]
    assert len(body["top_k"]) == 3
    assert body["inference_ms"] > 0


def test_predict_rejects_non_image_content_type(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_corrupt_image(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("fake.png", b"not really a png", "image/png")})
    assert response.status_code == 400


def test_predict_requires_file(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_predict_rejects_oversized_upload(service: InferenceService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MAX_IMAGE_BYTES", "100")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(service=service)) as small_limit_client:
            response = small_limit_client.post("/predict", files={"file": ("big.png", image_bytes(size=(256, 256)), "image/png")})
        assert response.status_code == 413
    finally:
        get_settings.cache_clear()
