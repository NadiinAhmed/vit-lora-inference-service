"""InferenceService: owns the loaded model and turns images into predictions.

Created once at API startup and reused for every request; loading a model per
request would add seconds of latency and memory churn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import ViTForImageClassification

from src.config import Settings
from src.model import load_quantized
from src.preprocessing import ImagePreprocessor


@dataclass(frozen=True)
class ClassScore:
    label: str
    confidence: float


class InferenceService:
    def __init__(self, model: ViTForImageClassification, preprocessor: ImagePreprocessor, top_k: int) -> None:
        self._model = model.eval()
        self._preprocessor = preprocessor
        self._labels: dict[int, str] = {int(i): name for i, name in model.config.id2label.items()}
        self._top_k = min(top_k, len(self._labels))

    @classmethod
    def from_settings(cls, settings: Settings) -> "InferenceService":
        directory = settings.quantized_model_dir
        if not (directory / "config.json").exists():
            raise FileNotFoundError(
                f"No quantized model found in '{directory}'. Run training/quantize.py first."
            )
        return cls(
            model=load_quantized(directory, settings.quantization_mode),
            preprocessor=ImagePreprocessor.from_pretrained(directory),
            top_k=settings.top_k,
        )

    @property
    def labels(self) -> list[str]:
        return [self._labels[i] for i in sorted(self._labels)]

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[ClassScore]:
        """Return the top-k classes, highest confidence first."""
        pixel_values = self._preprocessor([image])
        logits = self._model(pixel_values=pixel_values).logits[0]
        probabilities = logits.float().softmax(dim=-1)
        scores, indices = probabilities.topk(self._top_k)
        return [
            ClassScore(label=self._labels[int(i)], confidence=round(float(s), 4))
            for s, i in zip(scores, indices)
        ]
