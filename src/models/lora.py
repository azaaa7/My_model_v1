from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def parse_lora_layers(value: str | Iterable[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    value = str(value).strip()
    if not value or value.lower() == "all":
        return None
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_lora_targets(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def add_peft_lora_to_backbone(
    backbone: nn.Module,
    target_suffixes: tuple[str, ...],
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.1,
    lora_layers: list[int] | None = None,
) -> tuple[nn.Module, int]:
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the PEFT package. Install peft in the active environment."
        ) from exc

    config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        target_modules=list(target_suffixes),
        lora_dropout=float(dropout),
        bias="none",
    )
    if lora_layers:
        explicit_targets = []
        for idx in lora_layers:
            for suffix in target_suffixes:
                explicit_targets.append(f"blocks.{idx}.{suffix}")
        config.target_modules = explicit_targets

    backbone = inject_adapter_in_model(config, backbone)
    lora_count = sum(1 for _, module in backbone.named_modules() if hasattr(module, "lora_A"))
    return backbone, lora_count


def mark_only_lora_trainable(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def count_lora_parameters(module: nn.Module) -> int:
    return sum(param.numel() for name, param in module.named_parameters() if "lora_" in name and param.requires_grad)

