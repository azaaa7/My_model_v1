from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def mask_to_edge(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    mask = mask.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


class VideoMTLoss(nn.Module):
    """VidEoMT-style minimal mask objective: BCE + Dice + optional Edge BCE."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.bce_weight = float(cfg.get("bce_weight", 1.0))
        self.dice_weight = float(cfg.get("dice_weight", 1.0))
        self.edge_weight = float(cfg.get("edge_weight", 0.0))
        self.edge_kernel_size = int(cfg.get("edge_kernel_size", 5))
        self.use_pos_weight = bool(cfg.get("use_pos_weight", False))
        self.pos_weight_value = float(cfg.get("pos_weight", 1.0))
        self.dice_eps = float(cfg.get("dice_eps", 1.0e-6))

    @staticmethod
    def _extract_logits_and_aux(outputs, aux: dict[str, Any] | None):
        if isinstance(outputs, dict):
            return outputs["logits"], outputs.get("aux", aux or {})
        return outputs, aux or {}

    @staticmethod
    def _align_logits_masks(logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = masks.float().to(device=logits.device)
        if masks.ndim == 5:
            masks = masks.unsqueeze(3)
        if masks.ndim == 4:
            if logits.ndim == 6:
                logits = logits[:, logits.shape[1] // 2, logits.shape[2] // 2]
            elif logits.ndim == 5:
                logits = logits[:, logits.shape[1] // 2]
        elif masks.ndim == 5 and logits.ndim == 6:
            masks = masks[:, None]
        elif masks.ndim == 6 and logits.ndim == 5:
            logits = logits[:, None]

        if logits.shape[-2:] != masks.shape[-2:]:
            flat = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
            flat = F.interpolate(flat, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            logits = flat.reshape(*logits.shape[:-2], masks.shape[-2], masks.shape[-1])
        if masks.shape != logits.shape:
            try:
                masks = masks.reshape(*logits.shape)
            except RuntimeError as exc:
                raise RuntimeError(f"Cannot align masks {tuple(masks.shape)} to logits {tuple(logits.shape)}") from exc
        return logits, masks

    @staticmethod
    def _flatten_2d(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

    def _edge_loss(self, aux: dict[str, Any], masks_2d: torch.Tensor, logits_2d: torch.Tensor) -> torch.Tensor:
        edge_logits = None
        if isinstance(aux, dict):
            edge_logits = aux.get("edge_logits")
            if edge_logits is None:
                edge_logits = aux.get("boundary128")
        if self.edge_weight <= 0 or edge_logits is None or not torch.is_tensor(edge_logits):
            return logits_2d.sum() * 0.0
        edge_logits_2d = self._flatten_2d(edge_logits)
        edge_target = F.interpolate(masks_2d, size=edge_logits_2d.shape[-2:], mode="nearest")
        edge_target = mask_to_edge(edge_target, kernel_size=self.edge_kernel_size)
        return F.binary_cross_entropy_with_logits(edge_logits_2d, edge_target)

    def forward(
        self,
        outputs,
        targets: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del epoch
        logits, aux = self._extract_logits_and_aux(outputs, aux)
        logits, masks = self._align_logits_masks(logits, targets)
        logits_2d = self._flatten_2d(logits)
        masks_2d = self._flatten_2d(masks)

        pos_weight = None
        if self.use_pos_weight:
            pos_weight = torch.tensor([self.pos_weight_value], device=logits.device, dtype=logits.dtype)

        loss_bce = F.binary_cross_entropy_with_logits(logits_2d, masks_2d, pos_weight=pos_weight)
        loss_dice = dice_loss_from_logits(logits_2d, masks_2d, eps=self.dice_eps)
        if include_aux:
            loss_edge = self._edge_loss(aux, masks_2d, logits_2d)
        else:
            loss_edge = logits_2d.sum() * 0.0
        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice + self.edge_weight * loss_edge

        items = {
            "loss_total": float(total.detach().cpu()),
            "loss_bce": float(loss_bce.detach().cpu()),
            "loss_dice": float(loss_dice.detach().cpu()),
            "loss_edge": float(loss_edge.detach().cpu()),
            "main_loss": float(total.detach().cpu()),
        }
        return total, items
