"""Model quality gate: fail the pipeline if the pinned model is not good enough to ship.

Reads reports/quantization_report.json, which training/quantize.py writes together
with the model artifact, and checks the INT8 model (exactly what the API serves)
on the held-out test set.

Usage:
    python scripts/check_model_metrics.py --min-accuracy 0.85 --min-macro-f1 0.80 --max-accuracy-drop 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=Path("reports/quantization_report.json"))
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-macro-f1", type=float, default=0.80)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01,
                        help="Largest accuracy loss allowed from FP32 to INT8.")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    fp32, int8 = report["fp32"]["test"], report["int8"]["test"]
    drop = round(fp32["accuracy"] - int8["accuracy"], 4)

    checks = [
        (f"INT8 test accuracy {int8['accuracy']:.4f} >= {args.min_accuracy}",
         int8["accuracy"] >= args.min_accuracy),
        (f"INT8 test macro-F1 {int8['macro_f1']:.4f} >= {args.min_macro_f1}",
         int8["macro_f1"] >= args.min_macro_f1),
        (f"FP32 -> INT8 accuracy drop {drop:.4f} <= {args.max_accuracy_drop}",
         drop <= args.max_accuracy_drop),
    ]

    print(f"Model quality gate ({args.report}, {report['test_images']} test images)")
    for description, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")

    if all(passed for _, passed in checks):
        print("Model approved for packaging.")
        return 0
    print("Model rejected: fix the model or consciously change the thresholds in the workflow.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
