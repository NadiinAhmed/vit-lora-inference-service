"""Smoke test against the REAL trained + quantized model; skipped until it exists."""

import pytest
from PIL import Image

from src.config import get_settings
from src.inference import InferenceService

settings = get_settings()


@pytest.mark.skipif(not settings.quantized_weights_path.exists(), reason="Run training + quantization first.")
def test_real_quantized_model_predicts() -> None:
    service = InferenceService.from_settings(settings)
    assert len(service.labels) == 6
    scores = service.predict(Image.new("RGB", (512, 384), (120, 90, 60)))
    assert scores[0].label in service.labels
    assert 0.0 <= scores[0].confidence <= 1.0
