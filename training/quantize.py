"""Merge the best LoRA adapter into ViT, INT8-quantize it, measure the effect, save for serving.

Run from the repository root (after training):
    python -m training.quantize

Everything here runs on CPU on purpose: the served model runs on CPU, so size,
latency and accuracy are measured where they will actually be used.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save as safetensors_bytes
from torch.utils.data import DataLoader

from src.config import get_settings
from src.model import build_classifier, load_quantized, save_quantized
from src.preprocessing import ImagePreprocessor
from training.data import ImageClassificationDataset, download_trashnet, load_splits
from training.evaluation import evaluate

logger = logging.getLogger("quantize")


def fp32_size_mb(model: torch.nn.Module) -> float:
    """Size of the FP32 weights in the same safetensors format as the INT8 file."""
    state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    return round(len(safetensors_bytes(state)) / 1e6, 2)


@torch.inference_mode()
def measure_latency_ms(model: torch.nn.Module, pixel_values: torch.Tensor, runs: int) -> dict:
    """Single-image CPU latency. Warm-up runs are discarded (first calls allocate memory)."""
    model.eval()
    for _ in range(5):
        model(pixel_values=pixel_values)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model(pixel_values=pixel_values)
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return {"median_ms": round(statistics.median(timings), 2),
            "p95_ms": round(timings[int(0.95 * (len(timings) - 1))], 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    adapter_dir = settings.lora_adapter_dir
    meta = json.loads((adapter_dir / "training_meta.json").read_text())

    # 1. Rebuild the SAME test split used in training (same seed and fractions).
    data_root = args.data_root or download_trashnet(settings.data_dir)
    splits = load_splits(data_root, meta["val_fraction"], meta["test_fraction"], meta["seed"])
    assert splits.class_names == meta["class_names"], "Class names differ from training."
    preprocessor = ImagePreprocessor.from_pretrained(adapter_dir)
    test_loader = DataLoader(ImageClassificationDataset(splits.test, preprocessor), batch_size=args.batch_size)
    sample = next(iter(test_loader))["pixel_values"][:1]

    # 2. Merge LoRA into the base weights: W' = W + (alpha/r)·B·A. The result is a plain
    #    ViT, so inference no longer needs peft and has no adapter overhead.
    base = build_classifier(meta["base_model"], meta["class_names"])
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().eval()

    report: dict = {"hardware": {"cpu": platform.processor() or platform.machine(),
                                 "torch_threads": torch.get_num_threads()},
                    "storage": "INT8 per-row symmetric weights in safetensors (all nn.Linear layers)",
                    "runtime_mode": settings.quantization_mode,
                    "test_images": len(splits.test)}

    logger.info("Evaluating FP32 merged model on CPU ...")
    report["fp32"] = {"size_mb": fp32_size_mb(model),
                      "latency": measure_latency_ms(model, sample, args.latency_runs),
                      "test": evaluate(model, test_loader, "cpu", splits.class_names)}

    # 3. Save the serving artifact (INT8 weights, safetensors), then load it back exactly
    #    the way the API does. All INT8 numbers below are measured on this reloaded model,
    #    i.e. on precisely what gets served.
    out_dir = settings.quantized_model_dir
    weights_path = save_quantized(model, out_dir)
    preprocessor.save(out_dir)
    served = load_quantized(out_dir, settings.quantization_mode)
    logger.info("Evaluating INT8 model (as loaded for serving, mode=%s) on CPU ...", settings.quantization_mode)
    report["int8"] = {"size_mb": round(weights_path.stat().st_size / 1e6, 2),
                      "latency": measure_latency_ms(served, sample, args.latency_runs),
                      "test": evaluate(served, test_loader, "cpu", splits.class_names)}

    fp32, int8 = report["fp32"], report["int8"]
    report["comparison"] = {
        "size_reduction_x": round(fp32["size_mb"] / int8["size_mb"], 2),
        "latency_speedup_x": round(fp32["latency"]["median_ms"] / int8["latency"]["median_ms"], 2),
        "accuracy_change": round(int8["test"]["accuracy"] - fp32["test"]["accuracy"], 4),
        "macro_f1_change": round(int8["test"]["macro_f1"] - fp32["test"]["macro_f1"], 4),
    }
    settings.reports_dir.mkdir(exist_ok=True)
    (settings.reports_dir / "quantization_report.json").write_text(json.dumps(report, indent=2))
    logger.info("comparison=%s", report["comparison"])
    logger.info("Saved serving model to %s", out_dir)


if __name__ == "__main__":
    main()
