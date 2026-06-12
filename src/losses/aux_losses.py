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

        for key in (
            "ccm_mask32",
            "fgm_mask32",
            "noise_mask32",
            "adapter_mask32",
            "tcu_momentary_mask32",
            "tcu_gradual_mask32",
            "tcu_cumulative_mask32",
            "mask128",
        ):
            cfg = self.cfg.get(key, {}) or {}
            pred = aux.get(key)
            if not bool(cfg.get("enabled", False)) or pred is None:
                continue
            value = self._mask_loss(pred, target)
            total = total + float(cfg.get("weight", 1.0)) * value
            items[f"aux_{key}_loss"] = float(value.detach().cpu())
            alias = {
                "adapter_mask32": "adapter_mask32_loss",
                "tcu_momentary_mask32": "tcu_momentary_loss",
                "tcu_gradual_mask32": "tcu_gradual_loss",
                "tcu_cumulative_mask32": "tcu_cumulative_loss",
            }.get(key)
            if alias:
                items[alias] = float(value.detach().cpu())

        for key in ("boundary128", "noise_boundary", "adapter_boundary32"):
            cfg = self.cfg.get(key, {}) or {}
            pred = aux.get(key)
            if bool(cfg.get("enabled", False)) and pred is not None:
                value = self._boundary_loss(pred, target)
                total = total + float(cfg.get("weight", 1.0)) * value
                items[f"aux_{key}_loss"] = float(value.detach().cpu())
                if key == "adapter_boundary32":
                    items["adapter_boundary32_loss"] = float(value.detach().cpu())

        cfg = self.cfg.get("adapter_gate_l1", {}) or {}
        pred = aux.get("adapter_gate")
        if bool(cfg.get("enabled", False)) and pred is not None and torch.is_tensor(pred):
            value = pred.float().abs().mean()
            total = total + float(cfg.get("weight", 1.0)) * value
            items["aux_adapter_gate_l1_loss"] = float(value.detach().cpu())

        cfg = self.cfg.get("tcu_branch_diversity", {}) or {}
        maps = [aux.get(key) for key in ("tcu_momentary_mask32", "tcu_gradual_mask32", "tcu_cumulative_mask32")]
        if bool(cfg.get("enabled", False)) and all(torch.is_tensor(x) for x in maps):
            flats = [torch.sigmoid(x.float()).reshape(-1, x.shape[-3] * x.shape[-2] * x.shape[-1]) for x in maps]
            value = target.sum() * 0.0
            count = 0
            for i in range(len(flats)):
                for j in range(i + 1, len(flats)):
                    a = flats[i] - flats[i].mean(dim=1, keepdim=True)
                    b = flats[j] - flats[j].mean(dim=1, keepdim=True)
                    corr = (a * b).mean(dim=1) / (a.std(dim=1).clamp_min(1.0e-6) * b.std(dim=1).clamp_min(1.0e-6))
                    value = value + corr.abs().mean()
                    count += 1
            value = value / max(count, 1)
            total = total + float(cfg.get("weight", 1.0)) * value
            items["aux_tcu_branch_diversity_loss"] = float(value.detach().cpu())

        cfg = self.cfg.get("gate_regularization", {}) or {}
        pred = aux.get("gate_regularization")
        if bool(cfg.get("enabled", False)) and pred is not None and torch.is_tensor(pred):
            target_value = float(cfg.get("target", 0.0))
            value = (pred.float() - target_value).pow(2).mean()
            total = total + float(cfg.get("weight", 1.0)) * value
            items["aux_gate_regularization_loss"] = float(value.detach().cpu())

        items["aux_loss"] = float(total.detach().cpu())
        return total, items
