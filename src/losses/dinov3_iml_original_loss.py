from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 6:
        b, m, k, c, h, w = x.shape
        return x.reshape(b * m * k, c, h, w)
    if x.ndim == 5:
        if x.shape[2] == 1:
            b, k, c, h, w = x.shape
            return x.reshape(b * k, c, h, w)
        b, m, k, h, w = x.shape
        return x.reshape(b * m * k, 1, h, w)
    if x.ndim == 4:
        if x.shape[1] == 1:
            return x
        b, k, h, w = x.shape
        return x.reshape(b * k, 1, h, w)
    if x.ndim == 3:
        n, h, w = x.shape
        return x.reshape(n, 1, h, w)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def make_edge_mask(mask: torch.Tensor, width: int = 7) -> torch.Tensor:
    if width <= 0:
        return torch.zeros_like(mask)
    mask = (mask > 0.5).float()
    kernel = int(width)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


class DINOv3IMLOriginalLoss(nn.Module):
    """BCEWithLogits + edge-weighted BCEWithLogits from the original paper."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.edge_lambda = float(cfg.get("edge_lambda", 20.0))
        self.edge_mask_width = int(cfg.get("edge_mask_width", 7))
        reg_cfg = cfg.get("residual_regularization", {}) or {}
        self.residual_regularization_enabled = bool(reg_cfg.get("enabled", False))
        self.delta32_l1_weight = float(reg_cfg.get("delta32_l1_weight", 0.003))
        self.delta128_l1_weight = float(reg_cfg.get("delta128_l1_weight", 0.001))
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del epoch, include_aux
        logits = _flatten_to_nchw(logits)
        target = _flatten_to_nchw(target).float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        predict_loss = self.bce(logits, target)
        edge_mask = make_edge_mask(target, width=self.edge_mask_width)
        edge_loss = F.binary_cross_entropy_with_logits(logits, target, weight=edge_mask) * self.edge_lambda
        total = predict_loss + edge_loss
        items = {
            "predict_loss": float(predict_loss.detach().cpu()),
            "edge_loss": float(edge_loss.detach().cpu()),
        }
        if self.residual_regularization_enabled and aux:
            res_loss = logits.sum() * 0.0
            if aux.get("delta32") is not None:
                res_loss = res_loss + aux["delta32"].abs().mean() * self.delta32_l1_weight
            if aux.get("delta128") is not None:
                res_loss = res_loss + aux["delta128"].abs().mean() * self.delta128_l1_weight
            total = total + res_loss
            items["residual_regularization_loss"] = float(res_loss.detach().cpu())
        items["main_loss"] = float(total.detach().cpu())
        return total, items
