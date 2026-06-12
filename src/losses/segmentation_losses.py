from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_loss import BoundaryLoss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        target = target.float().flatten(1)
        inter = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        target = target.float().flatten(1)
        tp = (probs * target).sum(dim=1)
        fp = (probs * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - probs) * target).sum(dim=1)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tv.mean()


class TemporalDeltaLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 5:
            logits = logits[:, None]
            target = target[:, None]
        if logits.shape[2] <= 1:
            return logits.sum() * 0.0
        pred = torch.sigmoid(logits)
        return F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:].float() - target[:, :, :-1].float())


class SegmentationLoss(nn.Module):
    SUPPORTED = {"dice", "bce", "tversky", "boundary", "boundary_focal", "temporal_delta", "temporal_consistency"}

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.modules_map = nn.ModuleDict()
        self.weights: dict[str, float] = {}
        for name, args in cfg.items():
            name = str(name).lower()
            if name not in self.SUPPORTED:
                continue
            args = args or {}
            weight = float(args.get("weight", 1.0))
            if weight <= 0:
                continue
            self.weights[name] = weight
            if name == "dice":
                self.modules_map[name] = DiceLoss(float(args.get("smooth", 1e-6)))
            elif name == "tversky":
                self.modules_map[name] = TverskyLoss(
                    alpha=float(args.get("alpha", 0.3)),
                    beta=float(args.get("beta", 0.7)),
                    smooth=float(args.get("smooth", 1e-6)),
                )
            elif name in {"boundary", "boundary_focal"}:
                self.modules_map[name] = BoundaryLoss(kernel_size=int(args.get("kernel_size", 3)))
            elif name in {"temporal_delta", "temporal_consistency"}:
                self.modules_map[name] = TemporalDeltaLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target = target.float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits_4d = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
            logits_4d = F.interpolate(logits_4d, size=target.shape[-2:], mode="bilinear", align_corners=False)
            logits = logits_4d.reshape(*logits.shape[:-2], target.shape[-2], target.shape[-1])

        logits_4d = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
        target_4d = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1])
        total = logits.sum() * 0.0
        items: dict[str, float] = {}

        for name, weight in self.weights.items():
            if name == "bce":
                value = F.binary_cross_entropy_with_logits(logits_4d, target_4d)
            elif name in {"boundary", "boundary_focal", "temporal_delta", "temporal_consistency"}:
                value = self.modules_map[name](logits, target)
            else:
                value = self.modules_map[name](logits_4d, target_4d)
            total = total + weight * value
            items[f"{name}_loss"] = float(value.detach().cpu())

        items["main_loss"] = float(total.detach().cpu())
        return total, items
