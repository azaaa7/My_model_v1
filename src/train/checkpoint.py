from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _state_dict(model: nn.Module):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


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
    torch.save(
        {
            "model": _state_dict(model),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


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
