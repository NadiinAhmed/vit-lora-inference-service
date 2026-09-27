"""Model construction, INT8 quantization and serving-artifact save/load.

Only depends on transformers + torchao (no peft), so it is safe to import inside
the serving container. LoRA-specific code lives in ``training/``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, quantize_
from transformers import ViTConfig, ViTForImageClassification

WEIGHTS_FILENAME = "model_int8.safetensors"

QUANTIZATION_CONFIGS = {
    "weight_only": Int8WeightOnlyConfig,              # INT8 storage, float compute: portable
    "dynamic": Int8DynamicActivationInt8WeightConfig,  # INT8 compute: needs fast CPU int8 kernels
}


def build_classifier(base_model_name: str, class_names: list[str]) -> ViTForImageClassification:
    """Load the pretrained ViT backbone with a fresh classification head for our classes."""
    return ViTForImageClassification.from_pretrained(
        base_model_name,
        num_labels=len(class_names),
        id2label=dict(enumerate(class_names)),
        label2id={name: i for i, name in enumerate(class_names)},
        ignore_mismatched_sizes=True,  # replace any existing head with one sized for our classes
    )


def quantize_model(model: torch.nn.Module, mode: str = "weight_only") -> torch.nn.Module:
    """Quantize every nn.Linear in place (torchao).

    ViT's parameters and compute are almost entirely Linear layers (attention
    projections + MLP), so this captures most of the benefit without calibration data.
    """
    model.eval()
    quantize_(model, QUANTIZATION_CONFIGS[mode]())
    return model


def _linear_weight_keys(model: torch.nn.Module) -> set[str]:
    return {f"{name}.weight" for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


def _to_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row (per output channel) INT8: weight ~= int8 * scale."""
    scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    int8 = torch.round(weight / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return int8, scale


def save_quantized(model: ViTForImageClassification, directory: Path) -> Path:
    """Save config (incl. label names) + weights with every Linear weight stored as INT8.

    Stored as safetensors: raw tensors only, no pickled Python objects, so the file
    is portable across OSes/library versions and cannot execute code when loaded.
    Expects the FP32 (merged) model; returns the weights path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(str(directory))
    linear_keys = _linear_weight_keys(model)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        value = value.detach().float().cpu()
        if key in linear_keys:
            tensors[f"{key}.int8"], tensors[f"{key}.scale"] = _to_int8(value)
        else:
            tensors[key] = value.contiguous()  # biases, LayerNorms, embeddings: small, kept FP32
    weights_path = directory / WEIGHTS_FILENAME
    save_file(tensors, str(weights_path))
    return weights_path


def load_quantized(directory: Path, mode: str = "weight_only") -> ViTForImageClassification:
    """Rebuild ViT from config, restore the INT8-stored weights, then quantize in memory.

    The weights restored from INT8 already lie on the INT8 grid, so torchao's
    quantization reproduces them; the result is an INT8 model at runtime.
    """
    model = ViTForImageClassification(ViTConfig.from_pretrained(str(directory)))
    tensors = load_file(str(directory / WEIGHTS_FILENAME))
    state_dict = {
        key: (tensors[f"{key}.int8"].float() * tensors[f"{key}.scale"].unsqueeze(1))
        if f"{key}.int8" in tensors else tensors[key]
        for key in model.state_dict()
    }
    model.load_state_dict(state_dict)  # strict: any missing/unexpected key is an error
    return quantize_model(model, mode).eval()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
