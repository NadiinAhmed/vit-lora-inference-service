"""Monitoring report: compare production logs with the reference baseline.

Run from the repository root:
    python -m monitoring.report              # all logged predictions
    python -m monitoring.report --last 500   # only the most recent 500

Answers three questions and writes reports/monitoring/report.md (+ report.json):
  1. Data drift        - do incoming images look like the reference images?
  2. Prediction drift  - does the model predict classes and confidences like it used to?
  3. Performance       - on predictions with ground-truth feedback, is it still accurate?
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from monitoring.drift import ALERT, OK, WARNING, categorical_psi, ks_distance, numeric_psi, psi_status, worst_status
from src.config import get_settings
from src.image_stats import IMAGE_STAT_NAMES
from src.prediction_log import read_jsonl
from training.evaluation import compute_metrics

NOT_ENOUGH_DATA = "NOT ENOUGH DATA"

# Measured on the model-v1.0.0 reference: with no real drift, samples of 100 predictions
# still raised a false WARNING 86% of the time, 200 -> 12%, 300 -> 0%.
MIN_PREDICTIONS = 300
# With 100 labels an accuracy around 0.92 is known to roughly +/- 3 points; fewer is guesswork.
MIN_LABELS = 100
ACCURACY_DROP_WARNING, ACCURACY_DROP_ALERT = 0.05, 0.10
LOW_CONFIDENCE = 0.7  # a prediction below this is one the model is unsure about

_STATUS_ICON = {OK: "🟢", WARNING: "🟡", ALERT: "🔴", NOT_ENOUGH_DATA: "⚪"}


def build_report(reference: list[dict], predictions: list[dict], feedback: list[dict], model_version: str,
                 min_predictions: int = MIN_PREDICTIONS, min_labels: int = MIN_LABELS) -> dict:
    """Pure function: records in, report dict out (easy to test, no files involved)."""
    predictions = [p for p in predictions if p.get("model_version") == model_version]
    sections = {
        "data_drift": _data_drift(reference, predictions, min_predictions),
        "prediction_drift": _prediction_drift(reference, predictions, min_predictions),
        "performance": _performance(reference, predictions, feedback, min_labels),
    }
    judged = [s["status"] for s in sections.values() if s["status"] != NOT_ENOUGH_DATA]
    return {
        "status": worst_status(judged) if judged else NOT_ENOUGH_DATA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "reference_images": len(reference),
        "predictions": len(predictions),
        "period": [predictions[0]["timestamp"], predictions[-1]["timestamp"]] if predictions else None,
        "inference_ms_p50": _percentile([p["inference_ms"] for p in predictions], 50),
        "inference_ms_p95": _percentile([p["inference_ms"] for p in predictions], 95),
        **sections,
    }


def _data_drift(reference: list[dict], predictions: list[dict], min_predictions: int) -> dict:
    if len(predictions) < min_predictions:
        return {"status": NOT_ENOUGH_DATA, "reason": f"needs {min_predictions} predictions, has {len(predictions)}"}
    features = {name: _numeric_comparison(reference, predictions, name) for name in IMAGE_STAT_NAMES}
    return {"status": worst_status([f["status"] for f in features.values()]), "features": features}


def _prediction_drift(reference: list[dict], predictions: list[dict], min_predictions: int) -> dict:
    if len(predictions) < min_predictions:
        return {"status": NOT_ENOUGH_DATA, "reason": f"needs {min_predictions} predictions, has {len(predictions)}"}
    ref_labels, prod_labels = [r["label"] for r in reference], [p["label"] for p in predictions]
    class_psi = categorical_psi(ref_labels, prod_labels)
    confidence = _numeric_comparison(reference, predictions, "confidence")
    confidence["reference_low_share"] = _share(r["confidence"] < LOW_CONFIDENCE for r in reference)
    confidence["production_low_share"] = _share(p["confidence"] < LOW_CONFIDENCE for p in predictions)
    return {
        "status": worst_status([psi_status(class_psi), confidence["status"]]),
        "class_psi": class_psi,
        "class_status": psi_status(class_psi),
        "class_shares": {
            label: {"reference": _share(x == label for x in ref_labels),
                    "production": _share(x == label for x in prod_labels)}
            for label in sorted(set(ref_labels) | set(prod_labels))
        },
        "confidence": confidence,
    }


def _performance(reference: list[dict], predictions: list[dict], feedback: list[dict], min_labels: int) -> dict:
    class_names = sorted({r["true_label"] for r in reference})
    baseline = _metrics([r["label"] for r in reference], [r["true_label"] for r in reference], class_names)

    true_labels = {f["request_id"]: f["true_label"] for f in feedback}  # later feedback overrides earlier
    labelled = [(p["label"], true_labels[p["request_id"]]) for p in predictions if p["request_id"] in true_labels]
    coverage = round(len(labelled) / len(predictions), 4) if predictions else 0.0
    result = {"labelled_predictions": len(labelled), "label_coverage": coverage, "reference": baseline}
    if len(labelled) < min_labels:
        return {"status": NOT_ENOUGH_DATA, "reason": f"needs {min_labels} labelled predictions, has {len(labelled)}",
                **result}

    production = _metrics([pred for pred, _ in labelled], [true for _, true in labelled], class_names)
    accuracy_drop = round(baseline["accuracy"] - production["accuracy"], 4)
    status = ALERT if accuracy_drop > ACCURACY_DROP_ALERT else WARNING if accuracy_drop > ACCURACY_DROP_WARNING else OK
    return {"status": status, "accuracy_drop": accuracy_drop, "production": production, **result}


def _metrics(predicted: list[str], actual: list[str], class_names: list[str]) -> dict:
    """Accuracy, macro-F1 and per-class F1, computed by the same code as training evaluation."""
    index = {name: i for i, name in enumerate(class_names)}
    metrics = compute_metrics(torch.tensor([index[p] for p in predicted]),
                              torch.tensor([index[a] for a in actual]), class_names)
    return {k: metrics[k] for k in ("accuracy", "macro_f1", "per_class_f1")}


def _numeric_comparison(reference: list[dict], predictions: list[dict], field: str) -> dict:
    ref, prod = [r[field] for r in reference], [p[field] for p in predictions]
    psi = numeric_psi(ref, prod)
    return {"psi": psi, "ks": ks_distance(ref, prod), "status": psi_status(psi),
            "reference_mean": round(statistics.fmean(ref), 4), "production_mean": round(statistics.fmean(prod), 4)}


def _share(flags) -> float:
    """Fraction of True values, e.g. the share of predictions that are 'glass'."""
    flags = list(flags)
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def _percentile(values: list[float], percent: int) -> float | None:
    return round(float(np.percentile(values, percent)), 2) if values else None


# --- Markdown rendering -----------------------------------------------------

def icon(status: str) -> str:
    return f"{_STATUS_ICON[status]} {status}"


def render_markdown(report: dict) -> str:
    lines = [
        "# Model Monitoring Report", "",
        f"**Overall status: {icon(report['status'])}**", "",
        f"- Model version: `{report['model_version']}`",
        f"- Generated: {report['generated_at']}",
        f"- Predictions analysed: {report['predictions']} (reference: {report['reference_images']} test images)",
    ]
    if report["period"]:
        lines.append(f"- Period: {report['period'][0]} → {report['period'][1]}")
        lines.append(f"- Model latency: p50 {report['inference_ms_p50']} ms, p95 {report['inference_ms_p95']} ms")
    lines += ["", "PSI thresholds: < 0.10 OK, 0.10–0.25 WARNING, > 0.25 ALERT. "
                  "KS = largest gap between the two cumulative distributions (0–1).", ""]

    drift = report["data_drift"]
    lines += [f"## 1. Data drift — {icon(drift['status'])}", ""]
    if "features" in drift:
        lines += ["| Image statistic | Reference mean | Production mean | PSI | KS | Status |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {name} | {f['reference_mean']} | {f['production_mean']} | {f['psi']} | {f['ks']} | {icon(f['status'])} |"
                  for name, f in drift["features"].items()]
    else:
        lines.append(f"Not judged: {drift['reason']}.")

    pred = report["prediction_drift"]
    lines += ["", f"## 2. Prediction drift — {icon(pred['status'])}", ""]
    if "class_shares" in pred:
        conf = pred["confidence"]
        lines += [f"Predicted-class distribution: PSI {pred['class_psi']} ({icon(pred['class_status'])})", "",
                  "| Class | Reference share | Production share |", "|---|---|---|"]
        lines += [f"| {label} | {s['reference']:.1%} | {s['production']:.1%} |" for label, s in pred["class_shares"].items()]
        lines += ["", f"Confidence: PSI {conf['psi']}, KS {conf['ks']} ({icon(conf['status'])})", "",
                  "| | Reference | Production |", "|---|---|---|",
                  f"| Mean confidence | {conf['reference_mean']} | {conf['production_mean']} |",
                  f"| Share below {LOW_CONFIDENCE} | {conf['reference_low_share']:.1%} | {conf['production_low_share']:.1%} |"]
    else:
        lines.append(f"Not judged: {pred['reason']}.")

    perf = report["performance"]
    lines += ["", f"## 3. Model performance (ground-truth feedback) — {icon(perf['status'])}", "",
              f"Labelled predictions: {perf['labelled_predictions']} ({perf['label_coverage']:.1%} of predictions)", ""]
    if "production" in perf:
        ref, prod = perf["reference"], perf["production"]
        lines += [f"Accuracy drop vs reference: {perf['accuracy_drop']:+.4f} "
                  f"(WARNING > {ACCURACY_DROP_WARNING}, ALERT > {ACCURACY_DROP_ALERT})", "",
                  "| Metric | Reference | Production |", "|---|---|---|",
                  f"| Accuracy | {ref['accuracy']} | {prod['accuracy']} |",
                  f"| Macro-F1 | {ref['macro_f1']} | {prod['macro_f1']} |"]
        lines += [f"| F1 {name} | {ref['per_class_f1'][name]} | {prod['per_class_f1'][name]} |"
                  for name in ref["per_class_f1"]]
    else:
        lines.append(f"Not judged: {perf['reason']}. Without labels, sections 1–2 are the early warning.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", type=int, default=None, help="only analyse the most recent N predictions")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/monitoring")
    args = parser.parse_args()
    settings = get_settings()
    output_dir = args.output_dir or settings.reports_dir / "monitoring"

    reference = read_jsonl(settings.reports_dir / "monitoring_reference.jsonl")
    if not reference:
        raise SystemExit("No reference found. Run: python -m monitoring.build_reference")
    predictions = read_jsonl(settings.log_dir / "predictions.jsonl")
    if args.last:
        predictions = predictions[-args.last:]
    feedback = read_jsonl(settings.log_dir / "feedback.jsonl")

    report = build_report(reference, predictions, feedback, settings.model_version)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Overall status: {report['status']}  "
          f"(data drift: {report['data_drift']['status']}, prediction drift: {report['prediction_drift']['status']}, "
          f"performance: {report['performance']['status']})")
    print(f"Report written to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
