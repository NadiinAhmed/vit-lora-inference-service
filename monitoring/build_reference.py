"""Build the monitoring reference: how the served model behaves on images it never trained on.

Run once from the repository root (and again whenever a new model is released):
    python -m monitoring.build_reference

Drift means "different from normal", so "normal" must be recorded first. For every
image of the held-out TrashNet test split this records the same fields the live
service logs (image statistics, predicted label, confidence) plus the true label.

Why the test split and not the training split: the model has memorised its training
images, so its confidence on them is unrealistically high, and every real request
would then look like drift.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import get_settings
from src.image_stats import compute_image_stats
from src.inference import InferenceService
from src.preprocessing import open_image
from training.data import Sample, download_trashnet, load_splits

logger = logging.getLogger("build_reference")


def build_reference_records(samples: list[Sample], class_names: list[str],
                            service: InferenceService) -> list[dict]:
    """One record per image, with the same fields as a line of logs/predictions.jsonl."""
    records = []
    for index, (path, label_id) in enumerate(samples, start=1):
        image = open_image(path)
        top = service.predict(image)[0]
        records.append({
            "image": f"{path.parent.name}/{path.name}",
            "true_label": class_names[label_id],
            "label": top.label,
            "confidence": top.confidence,
            "width": image.size[0],
            "height": image.size[1],
            **compute_image_stats(image),
        })
        if index % 50 == 0:
            logger.info("processed %d/%d images", index, len(samples))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None, help="TrashNet class folders (default: download)")
    parser.add_argument("--output", type=Path, default=None, help="default: reports/monitoring_reference.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    output = args.output or settings.reports_dir / "monitoring_reference.jsonl"

    # Rebuild exactly the test split used in training (seed and fractions from the committed summary).
    summary = json.loads((settings.reports_dir / "training_summary.json").read_text())
    data_root = args.data_root or download_trashnet(settings.data_dir)
    splits = load_splits(data_root, summary["val_fraction"], summary["test_fraction"], summary["seed"])

    service = InferenceService.from_settings(settings)  # the served INT8 model, loaded like the API does
    if splits.class_names != service.labels:
        raise SystemExit(f"Dataset classes {splits.class_names} differ from model classes {service.labels}.")

    logger.info("Building reference from %d test images ...", len(splits.test))
    records = build_reference_records(splits.test, splits.class_names, service)
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    accuracy = sum(r["label"] == r["true_label"] for r in records) / len(records)
    logger.info("Saved %d records to %s (model_version=%s, accuracy=%.4f)",
                len(records), output, settings.model_version, accuracy)

    # Sanity check: the baseline must reproduce the accuracy measured when the model was built.
    expected = json.loads((settings.reports_dir / "quantization_report.json").read_text())["int8"]["test"]["accuracy"]
    if abs(accuracy - expected) > 0.005:
        logger.warning("Accuracy %.4f differs from quantization_report.json (%.4f): "
                       "is this the same model and test split?", accuracy, expected)
    else:
        logger.info("Accuracy matches quantization_report.json (%.4f)", expected)


if __name__ == "__main__":
    main()
