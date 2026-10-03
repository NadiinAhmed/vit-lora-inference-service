"""The simulator's image change: darkening lowers brightness without changing the image shape."""

from PIL import Image

from monitoring.simulate_traffic import darken
from src.image_stats import compute_image_stats


def test_darken_lowers_brightness_only() -> None:
    image = Image.new("RGB", (120, 90), (200, 150, 100))
    before, after = compute_image_stats(image), compute_image_stats(darken(image))
    assert after["brightness"] < before["brightness"] * 0.5
    assert after["aspect_ratio"] == before["aspect_ratio"]
