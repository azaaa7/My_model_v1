from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _inner_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _is_frozen_backbone_key(name: str) -> bool:
    return name.startswith("encoder.backbone.") and "lora_" not in name


def _state_dict(model: nn.Module, compact: bool = False):
    model = _inner_model(model)
    state = model.state_dict()
    if not compact:
        return state

    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    compact_state = {}
    for name, tensor in state.items():
        if name in trainable or not _is_frozen_backbone_key(name):
            compact_state[name] = tensor
    return compact_state


def _checkpoint_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg.get("train", {}) or {}
    merged: dict[str, Any] = {}
    if isinstance(cfg.get("checkpoint"), dict):
        merged.update(cfg.get("checkpoint", {}) or {})
    if isinstance(train_cfg.get("checkpoint"), dict):
        merged.update(train_cfg.get("checkpoint", {}) or {})
    return merged


def _uses_frozen_dinov3_backbone(cfg: dict[str, Any]) -> bool:
    dinov3_cfg = cfg.get("dinov3", {}) or {}
    return bool(dinov3_cfg.get("freeze_backbone", False))


def _should_save_compact(cfg: dict[str, Any], ckpt_cfg: dict[str, Any]) -> bool:
    explicit = ckpt_cfg.get("save_trainable_only", ckpt_cfg.get("compact"))
    if explicit is not None:
        return bool(explicit)
    # Default to compact checkpoints when the pretrained DINOv3 backbone is frozen.
    # The backbone will be reconstructed from the config's local pretrained weights
    # before loading the trainable deltas from the checkpoint.
    return _uses_frozen_dinov3_backbone(cfg)


def _materialize_state_dict_for_load(target: nn.Module, state: dict[str, Any], ckpt: dict[str, Any]) -> dict[str, Any]:
    fmt = str(ckpt.get("checkpoint_format", ""))
    is_compact = fmt == "compact_trainable"
    if not is_compact:
        target_keys = set(target.state_dict().keys())
        state_keys = set(state.keys())
        is_compact = bool(target_keys - state_keys)
    if not is_compact:
        return state

    merged = target.state_dict()
    merged.update(state)
    return merged


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = _checkpoint_cfg(cfg)
    compact = _should_save_compact(cfg, ckpt_cfg)
    save_optimizer = bool(ckpt_cfg.get("save_optimizer", True))
    save_scheduler = bool(ckpt_cfg.get("save_scheduler", True))
    payload = {
        "model": _state_dict(model, compact=compact),
        "optimizer": optimizer.state_dict() if optimizer is not None and save_optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler is not None and save_scheduler else None,
        "epoch": int(epoch),
        "metrics": metrics,
        "config": cfg,
        "checkpoint_format": "compact_trainable" if compact else "full",
    }
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    target = model.module if hasattr(model, "module") else model
    if isinstance(state, dict):
        state = _materialize_state_dict_for_load(target, state, ckpt if isinstance(ckpt, dict) else {})
    missing, unexpected = target.load_state_dict(state, strict=strict)
    print(f"[checkpoint] loaded {path} missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"[checkpoint] skipped optimizer state: {exc}")
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError as exc:
            print(f"[checkpoint] skipped scheduler state: {exc}")
    return ckpt
