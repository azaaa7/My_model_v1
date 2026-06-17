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
    compact = bool(ckpt_cfg.get("save_trainable_only", ckpt_cfg.get("compact", False)))
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
