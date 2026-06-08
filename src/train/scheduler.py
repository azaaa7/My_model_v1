from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_scheduler(optimizer: Optimizer, cfg: dict[str, Any]):
    sched_cfg = cfg.get("scheduler", {}) or {}
    epochs = int((cfg.get("train", {}) or {}).get("n_epochs", 100))
    warmup = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    if str(sched_cfg.get("type", "cosine")).lower() != "cosine":
        return None
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=min_lr)
    if warmup <= 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])

