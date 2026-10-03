"""Image statistics on images whose correct answer is known in advance."""

import pytest
from PIL import Image

from src.image_stats import IMAGE_STAT_NAMES, compute_image_stats


def test_returns_every_stat() -> None:
    assert tuple(compute_image_stats(Image.new("RGB", (64, 64))).keys()) == IMAGE_STAT_NAMES


def test_brightness_of_black_and_white() -> None:
    assert compute_image_stats(Image.new("RGB", (64, 64), "black"))["brightness"] == 0.0
    assert compute_image_stats(Image.new("RGB", (64, 64), "white"))["brightness"] == 1.0


def test_contrast_of_flat_and_half_black_half_white_images() -> None:
    assert compute_image_stats(Image.new("RGB", (64, 64), (128, 128, 128)))["contrast"] == 0.0

    split = Image.new("RGB", (128, 128), "black")
    split.paste((255, 255, 255), (0, 0, 64, 128))  # left half white
    assert compute_image_stats(split)["contrast"] == pytest.approx(0.5, abs=0.01)


def test_saturation_of_grey_and_pure_red() -> None:
    assert compute_image_stats(Image.new("RGB", (64, 64), (90, 90, 90)))["saturation"] == 0.0
    assert compute_image_stats(Image.new("RGB", (64, 64), (255, 0, 0)))["saturation"] == 1.0


def test_aspect_ratio_uses_original_size() -> None:
    assert compute_image_stats(Image.new("RGB", (200, 100)))["aspect_ratio"] == 2.0
