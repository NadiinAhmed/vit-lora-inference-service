"""The reference builder produces one record per image with the same fields as live logs."""

from pathlib import Path

from PIL import Image

from monitoring.build_reference import build_reference_records
from src.inference import InferenceService
from tests.conftest import CLASS_NAMES


def test_one_record_per_image_with_live_log_fields(service: InferenceService, tmp_path: Path) -> None:
    glass_dir = tmp_path / "glass"
    glass_dir.mkdir()
    samples = []
    for i, colour in enumerate(["white", "black"]):
        path = glass_dir / f"glass{i}.jpg"
        Image.new("RGB", (100, 50), colour).save(path)
        samples.append((path, CLASS_NAMES.index("glass")))

    records = build_reference_records(samples, CLASS_NAMES, service)

    assert len(records) == 2
    assert records[0]["image"] == "glass/glass0.jpg"
    assert all(r["true_label"] == "glass" and r["label"] in CLASS_NAMES for r in records)
    assert records[0]["brightness"] == 1.0 and records[1]["brightness"] == 0.0
    assert records[0]["aspect_ratio"] == 2.0
    # Same field names as logs/predictions.jsonl, so Step 6 can compare like with like.
    assert {"label", "confidence", "width", "height", "brightness", "contrast", "saturation"} <= records[0].keys()
