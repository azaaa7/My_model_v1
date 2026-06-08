from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_loss import boundary_target
from .segmentation_losses import DiceLoss


class AuxiliaryLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self.dice = DiceLoss()

    @staticmethod
    def _align_pred_target(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim == 4:
            # Single-clip train target is only the center frame [B,1,H,W].
            if pred.ndim == 6:
                pred = pred[:, pred.shape[1] // 2, pred.shape[2] // 2]
            elif pred.ndim == 5:
                pred = pred[:, pred.shape[1] // 2]
        elif target.ndim == 5 and pred.ndim == 6:
            # Single-clip eval target [B,K,1,H,W] vs pred [B,1,K,1,H,W].
            target = target[:, None]
        return pred, target

    @staticmethod
    def _resize_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        flat = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1]).float()
        flat = F.interpolate(flat, size=size, mode="nearest")
        return flat

    def _mask_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._align_pred_target(pred, target)
        target_rs = self._resize_target(target, pred.shape[-2:])
        pred_4d = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        return F.binary_cross_entropy_with_logits(pred_4d, target_rs) + self.dice(pred_4d, target_rs)

    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._align_pred_target(pred, target)
        target_rs = self._resize_target(target, pred.shape[-2:])
        edge = boundary_target(target_rs)
        pred_4d = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        return F.binary_cross_entropy_with_logits(pred_4d, edge)

    def forward(self, aux: dict[str, Any] | None, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not aux:
            return target.sum() * 0.0, {}
        total = target.sum() * 0.0
        items: dict[str, float] = {}

        for key in ("ccm_mask32", "fgm_mask32", "mask128"):
            cfg = self.cfg.get(key, {}) or {}
            pred = aux.get(key)
            if not bool(cfg.get("enabled", False)) or pred is None:
                continue
            value = self._mask_loss(pred, target)
            total = total + float(cfg.get("weight", 1.0)) * value
            items[f"aux_{key}_loss"] = float(value.detach().cpu())

        cfg = self.cfg.get("boundary128", {}) or {}
        pred = aux.get("boundary128")
        if bool(cfg.get("enabled", False)) and pred is not None:
            value = self._boundary_loss(pred, target)
            total = total + float(cfg.get("weight", 1.0)) * value
            items["aux_boundary128_loss"] = float(value.detach().cpu())

        items["aux_loss"] = float(total.detach().cpu())
        return total, items
