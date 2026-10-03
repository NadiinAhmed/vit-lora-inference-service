"""Simple, explainable statistics that describe an input image.

Tabular models have columns (age, country...) whose distributions can be compared
directly. Images do not, so each image is summarised by a few numbers a person can
reason about: is it darker, flatter, less colourful, a different shape?

The same function runs on the reference dataset and on live requests, so both are
measured identically (the same anti-skew rule as src/preprocessing.py).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGE_STAT_NAMES = ("brightness", "contrast", "saturation", "aspect_ratio")

# Stats are computed on a small copy: fast, and independent of the upload's resolution.
_ANALYSIS_SIZE = (128, 128)


def compute_image_stats(image: Image.Image) -> dict[str, float]:
    """Return brightness, contrast and saturation (all 0..1) and the width/height ratio.

    brightness   - mean grey level: lighting conditions (dark warehouse vs studio)
    contrast     - spread of grey levels: flat/foggy vs crisp images
    saturation   - mean colour intensity: background or camera colour changes
    aspect_ratio - width / height: a different camera or cropping
    """
    width, height = image.size
    small = image.convert("RGB").resize(_ANALYSIS_SIZE)
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    saturation = np.asarray(small.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    return {
        "brightness": round(float(gray.mean()), 4),
        "contrast": round(float(gray.std()), 4),
        "saturation": round(float(saturation.mean()), 4),
        "aspect_ratio": round(width / height, 4),
    }
