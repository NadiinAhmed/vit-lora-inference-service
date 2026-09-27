"""Fine-tune ViT with LoRA, keep the best checkpoint (by validation macro-F1), evaluate on test.

Run from the repository root:
    python -m training.train_lora --epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

from src.config import get_settings
from src.model import build_classifier, count_parameters
from src.preprocessing import ImagePreprocessor
from training.data import ImageClassificationDataset, download_trashnet, load_splits
from training.evaluation import evaluate

logger = logging.getLogger("train_lora")

# LoRA adapters go on the attention query and value projections (the choice in the
# original LoRA paper). The new classifier head has no pretrained weights, so it is
# trained fully ("modules_to_save") instead of through a low-rank adapter.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_FULLY_TRAINED_MODULES = ["classifier"]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=settings.base_model_name)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Folder with one sub-folder per class. Default: download TrashNet.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 mixed precision on GPU.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ImageClassificationDataset, batch_size: int, shuffle: bool, workers: int, device: str
                ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device == "cuda", persistent_workers=workers > 0)


def linear_warmup_decay(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
                        ) -> torch.optim.lr_scheduler.LambdaLR:
    """LR rises linearly during warmup (stabilises the new head), then decays linearly to 0."""
    warmup = max(1, math.ceil(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    logger.info("device=%s mixed_precision_fp16=%s", device, use_amp)

    # ---- Data --------------------------------------------------------------
    data_root = args.data_root or download_trashnet(settings.data_dir)
    splits = load_splits(data_root, args.val_fraction, args.test_fraction, args.seed)
    logger.info("classes=%s train=%d val=%d test=%d", splits.class_names,
                len(splits.train), len(splits.validation), len(splits.test))

    preprocessor = ImagePreprocessor.from_pretrained(args.base_model)
    train_loader = make_loader(ImageClassificationDataset(splits.train, preprocessor, augment=True),
                               args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(ImageClassificationDataset(splits.validation, preprocessor),
                             args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(ImageClassificationDataset(splits.test, preprocessor),
                              args.batch_size, False, args.num_workers, device)

    # ---- Model + LoRA ------------------------------------------------------
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=LORA_TARGET_MODULES, modules_to_save=LORA_FULLY_TRAINED_MODULES)
    model = get_peft_model(build_classifier(args.base_model, splits.class_names), lora_config).to(device)
    trainable, total = count_parameters(model)
    logger.info("trainable_params=%d total_params=%d trainable_pct=%.2f%%", trainable, total, 100 * trainable / total)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = linear_warmup_decay(optimizer, args.epochs * len(train_loader), args.warmup_ratio)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)  # prevents FP16 gradient underflow

    # ---- Training loop -----------------------------------------------------
    adapter_dir = settings.lora_adapter_dir
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running_loss = time.perf_counter(), 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        val = evaluate(model, val_loader, device, splits.class_names)
        record = {"epoch": epoch, "train_loss": round(running_loss / len(train_loader), 4),
                  "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                  "epoch_seconds": round(time.perf_counter() - start, 1)}
        history.append(record)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            model.save_pretrained(str(adapter_dir))  # adapter + classifier head only
            preprocessor.save(adapter_dir)
        logger.info("%s%s", record, "  <- best, saved" if improved else "")

    # ---- Test evaluation of the BEST checkpoint (not the last epoch) ---------
    best_model = PeftModel.from_pretrained(build_classifier(args.base_model, splits.class_names),
                                           str(adapter_dir)).to(device)
    test = evaluate(best_model, test_loader, device, splits.class_names)
    logger.info("best_epoch=%d test_accuracy=%.4f test_macro_f1=%.4f", best_epoch, test["accuracy"], test["macro_f1"])

    meta = {"base_model": args.base_model, "class_names": splits.class_names, "seed": args.seed,
            "val_fraction": args.val_fraction, "test_fraction": args.test_fraction}
    (adapter_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    summary = {
        **meta,
        "device": device, "mixed_precision_fp16": use_amp,
        "split_sizes": {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)},
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in {"base_model", "data_root"}},
        "lora": {"target_modules": LORA_TARGET_MODULES, "fully_trained_modules": LORA_FULLY_TRAINED_MODULES},
        "parameters": {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3)},
        "adapter_size_mb": round(sum(f.stat().st_size for f in adapter_dir.glob("*.safetensors")) / 1e6, 2),
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history, "test": test,
    }
    settings.reports_dir.mkdir(exist_ok=True)
    report_path = settings.reports_dir / "training_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved best adapter to %s and report to %s", adapter_dir, report_path)


if __name__ == "__main__":
    main()
