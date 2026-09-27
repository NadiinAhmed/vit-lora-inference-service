"""Image decoding and preprocessing shared by training AND serving.

Both sides import this module, so the model always sees images prepared the same
way (orientation, colour mode, resize, normalisation). Any difference between
training-time and serving-time preprocessing ("training/serving skew") silently
lowers accuracy in production.
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import ViTImageProcessorPil


class InvalidImageError(ValueError):
    """Raised when bytes cannot be decoded into a usable image."""


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply EXIF rotation (phone photos) and force 3-channel RGB."""
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def decode_image(data: bytes) -> Image.Image:
    """Decode untrusted uploaded bytes into a normalised RGB image."""
    if not data:
        raise InvalidImageError("Empty file.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # cheap integrity check before decoding all pixels
        image = Image.open(io.BytesIO(data))  # verify() invalidates the object; reopen
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("File is not a valid image.") from exc
    return normalize_image(image)


def open_image(path: Path) -> Image.Image:
    """Load an image from disk (training) with the same normalisation as serving."""
    with Image.open(path) as image:
        image.load()
        return normalize_image(image)


class ImagePreprocessor:
    """Turns PIL images into the normalised tensor batch ViT expects.

    Wraps the Pillow-backed Hugging Face processor explicitly, so training and
    serving can never end up on different resize backends.
    """

    def __init__(self, processor: ViTImageProcessorPil) -> None:
        self._processor = processor

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path) -> "ImagePreprocessor":
        return cls(ViTImageProcessorPil.from_pretrained(str(name_or_path)))

    def save(self, directory: Path) -> None:
        self._processor.save_pretrained(str(directory))

    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        """Return pixel values with shape (batch, 3, height, width)."""
        return self._processor(images=images, return_tensors="pt")["pixel_values"]
