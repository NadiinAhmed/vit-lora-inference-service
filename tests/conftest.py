"""Shared fixtures: a tiny, randomly initialised ViT with the real architecture.

Tests exercise the exact serving code path (quantize -> save -> load -> predict)
without downloading the 86M-parameter model, so they run in seconds and offline.
"""

import io
from pathlib import Path

import pytest
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessorPil

from src.inference import InferenceService
from src.model import load_quantized, save_quantized
from src.preprocessing import ImagePreprocessor

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=64, num_labels=len(CLASS_NAMES),
                       id2label=dict(enumerate(CLASS_NAMES)),
                       label2id={n: i for i, n in enumerate(CLASS_NAMES)})
    directory = tmp_path_factory.mktemp("quantized")
    save_quantized(ViTForImageClassification(config), directory)  # FP32 in, INT8 on disk
    ImagePreprocessor(ViTImageProcessorPil(size={"height": 32, "width": 32})).save(directory)
    return directory


@pytest.fixture(scope="session")
def service(tiny_model_dir: Path) -> InferenceService:
    return InferenceService(load_quantized(tiny_model_dir), ImagePreprocessor.from_pretrained(tiny_model_dir), top_k=3)


def image_bytes(mode: str = "RGB", fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=fmt)
    return buffer.getvalue()
