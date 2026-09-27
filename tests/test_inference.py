"""Preprocessing, quantized save/load and InferenceService behaviour."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.inference import InferenceService
from safetensors.torch import load_file

from src.model import WEIGHTS_FILENAME, load_quantized
from src.preprocessing import ImagePreprocessor, InvalidImageError, decode_image
from tests.conftest import CLASS_NAMES, image_bytes


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_decode_converts_any_mode_to_rgb(mode: str) -> None:
    assert decode_image(image_bytes(mode)).mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"definitely not an image", image_bytes()[:40]])
def test_decode_rejects_invalid_bytes(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_preprocessor_output_shape(tiny_model_dir: Path) -> None:
    preprocessor = ImagePreprocessor.from_pretrained(tiny_model_dir)
    pixels = preprocessor([Image.new("RGB", (100, 60)), Image.new("RGB", (20, 20))])
    assert pixels.shape == (2, 3, 32, 32)


def test_saved_linear_weights_are_int8(tiny_model_dir: Path) -> None:
    tensors = load_file(str(tiny_model_dir / WEIGHTS_FILENAME))
    int8_keys = [k for k in tensors if k.endswith(".int8")]
    assert int8_keys, "expected INT8-stored Linear weights"
    assert all(tensors[k].dtype == torch.int8 for k in int8_keys)
    assert not any(k.endswith("query.weight") or k.endswith("q_proj.weight") for k in tensors)  # no FP32 copies


def test_quantized_model_reload_is_deterministic(tiny_model_dir: Path) -> None:
    pixels = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        first = load_quantized(tiny_model_dir)(pixel_values=pixels).logits
        second = load_quantized(tiny_model_dir)(pixel_values=pixels).logits
    assert torch.equal(first, second)


def test_predict_returns_sorted_top_k(service: InferenceService) -> None:
    scores = service.predict(Image.new("RGB", (64, 64), (200, 30, 30)))
    assert len(scores) == 3
    assert all(s.label in CLASS_NAMES for s in scores)
    confidences = [s.confidence for s in scores]
    assert confidences == sorted(confidences, reverse=True)
    assert 0.0 < sum(confidences) <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", ["weight_only", "dynamic"])
def test_both_quantization_modes_load_and_predict(tiny_model_dir: Path, mode: str) -> None:
    svc = InferenceService(load_quantized(tiny_model_dir, mode), ImagePreprocessor.from_pretrained(tiny_model_dir), top_k=3)
    assert svc.predict(Image.new("RGB", (32, 32)))[0].label in CLASS_NAMES


def test_top_k_is_capped_at_number_of_classes(tiny_model_dir: Path) -> None:
    svc = InferenceService(load_quantized(tiny_model_dir), ImagePreprocessor.from_pretrained(tiny_model_dir), top_k=50)
    assert len(svc.predict(Image.new("RGB", (32, 32)))) == len(CLASS_NAMES)
