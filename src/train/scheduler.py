from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def build_scheduler(optimizer: Optimizer, cfg: dict[str, Any]):
    sched_cfg = cfg.get("scheduler", {}) or {}
    epochs = int((cfg.get("train", {}) or {}).get("n_epochs", 100))
    warmup = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    sched_type = str(sched_cfg.get("type", "cosine")).lower()
    if sched_type == "poly":
        power = float(sched_cfg.get("power", 0.9))
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

        def make_lambda(base_lr: float):
            def lr_lambda(epoch: int):
                if warmup > 0 and epoch < warmup:
                    factor = float(epoch + 1) / float(warmup)
                else:
                    denom = max(1, epochs - warmup)
                    progress = min(max((epoch - warmup) / denom, 0.0), 1.0)
                    factor = (1.0 - progress) ** power
                if base_lr > 0:
                    factor = max(min_lr / base_lr, factor)
                return factor

            return lr_lambda

        return LambdaLR(optimizer, lr_lambda=[make_lambda(lr) for lr in base_lrs])
    if sched_type != "cosine":
        return None
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=min_lr)
    if warmup <= 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])
