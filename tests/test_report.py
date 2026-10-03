"""Monitoring report: section statuses, minimum-data rules and label joining."""

from monitoring.drift import ALERT, OK
from monitoring.report import NOT_ENOUGH_DATA, build_report, render_markdown

VERSION = "model-v1.0.0"


def _reference(n: int = 40) -> list[dict]:
    labels = ["glass", "paper"]
    return [{"true_label": labels[i % 2], "label": labels[i % 2], "confidence": 0.90 + (i % 10) / 100,
             "brightness": 0.5 + (i % 10) / 50, "contrast": 0.2, "saturation": 0.3, "aspect_ratio": 1.3333}
            for i in range(n)]


def _predictions(reference: list[dict], version: str = VERSION, **overrides) -> list[dict]:
    return [{**r, **overrides, "request_id": f"id{i}", "model_version": version,
             "timestamp": f"2026-10-03T10:00:{i % 60:02d}+00:00", "inference_ms": 250.0}
            for i, r in enumerate(reference)]


def _feedback(predictions: list[dict], wrong: int = 0) -> list[dict]:
    """True labels for every prediction; the first `wrong` predictions are marked as mistakes."""
    flip = {"glass": "paper", "paper": "glass"}
    return [{"request_id": p["request_id"], "true_label": flip[p["label"]] if i < wrong else p["label"]}
            for i, p in enumerate(predictions)]


def test_production_like_reference_is_ok() -> None:
    reference = _reference()
    predictions = _predictions(reference)
    report = build_report(reference, predictions, _feedback(predictions), VERSION, min_predictions=10, min_labels=10)
    assert report["status"] == OK
    assert report["data_drift"]["features"]["brightness"]["psi"] == 0.0
    assert report["performance"]["production"]["accuracy"] == 1.0


def test_darker_images_less_confidence_and_mistakes_raise_alerts() -> None:
    reference = _reference()
    predictions = _predictions(reference, brightness=0.2, confidence=0.5, label="glass")
    report = build_report(reference, predictions, _feedback(predictions, wrong=20), VERSION,
                          min_predictions=10, min_labels=10)
    assert report["data_drift"]["features"]["brightness"]["status"] == ALERT
    assert report["data_drift"]["features"]["contrast"]["status"] == OK  # unchanged features stay OK
    assert report["prediction_drift"]["class_status"] == ALERT
    assert report["prediction_drift"]["confidence"]["status"] == ALERT
    assert report["performance"]["status"] == ALERT
    assert report["status"] == ALERT


def test_too_few_predictions_are_not_judged() -> None:
    reference = _reference()
    predictions = _predictions(reference)[:5]
    report = build_report(reference, predictions, [], VERSION, min_predictions=10, min_labels=10)
    assert report["data_drift"]["status"] == NOT_ENOUGH_DATA
    assert report["prediction_drift"]["status"] == NOT_ENOUGH_DATA
    assert report["performance"]["status"] == NOT_ENOUGH_DATA
    assert report["status"] == NOT_ENOUGH_DATA


def test_drift_is_judged_even_without_labels() -> None:
    reference = _reference()
    report = build_report(reference, _predictions(reference), [], VERSION, min_predictions=10, min_labels=10)
    assert report["performance"]["status"] == NOT_ENOUGH_DATA
    assert report["performance"]["labelled_predictions"] == 0
    assert report["status"] == OK  # decided by the sections that do have enough data


def test_only_the_current_model_version_is_analysed() -> None:
    reference = _reference()
    old_model_traffic = _predictions(reference, version="model-v0.9.0", brightness=0.0)
    report = build_report(reference, old_model_traffic + _predictions(reference), [], VERSION, min_predictions=10)
    assert report["predictions"] == len(reference)
    assert report["data_drift"]["status"] == OK


def test_latest_feedback_for_a_prediction_wins() -> None:
    reference = _reference()
    predictions = _predictions(reference)
    corrected = _feedback(predictions, wrong=len(predictions)) + _feedback(predictions)  # all wrong, then all fixed
    report = build_report(reference, predictions, corrected, VERSION, min_predictions=10, min_labels=10)
    assert report["performance"]["production"]["accuracy"] == 1.0


def test_markdown_contains_every_section() -> None:
    reference = _reference()
    for predictions in (_predictions(reference), []):
        markdown = render_markdown(build_report(reference, predictions, [], VERSION, min_predictions=10))
        for heading in ("Overall status", "## 1. Data drift", "## 2. Prediction drift", "## 3. Model performance"):
            assert heading in markdown
