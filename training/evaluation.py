"""Evaluation shared by training (checkpoint selection) and quantization (before/after).

Macro-F1 is reported next to accuracy because TrashNet is imbalanced: a model can
score well on accuracy while failing the rare 'trash' class; macro-F1 weights
every class equally and exposes that.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict:
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)  # rows: true, cols: predicted
    for true, pred in zip(labels.tolist(), predictions.tolist()):
        confusion[true, pred] += 1

    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp(min=1)
    recall = true_positive / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "accuracy": round(true_positive.sum().item() / max(len(labels), 1), 4),
        "macro_f1": round(f1.mean().item(), 4),
        "per_class_f1": {name: round(v, 4) for name, v in zip(class_names, f1.tolist())},
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, class_names: list[str]) -> dict:
    """Run the model over a loader; return metrics plus mean cross-entropy loss."""
    model.eval()
    all_predictions, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits.float()
        total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        all_predictions.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    labels_tensor = torch.cat(all_labels)
    metrics = compute_metrics(torch.cat(all_predictions), labels_tensor, class_names)
    metrics["loss"] = round(total_loss / max(len(labels_tensor), 1), 4)
    return metrics
