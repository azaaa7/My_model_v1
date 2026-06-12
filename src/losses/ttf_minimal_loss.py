from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ramp_weight(base_weight: float, epoch: int | None, warmup_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if epoch is None or warmup_epochs <= 0:
        return float(base_weight)
    return float(base_weight) * min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


class TTFMinimalLoss(nn.Module):
    """Minimal TTF objective: segmentation loss + temporal difference alignment."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self.seg_cfg = self.cfg.get("seg", {}) or {}
        self.focal_cfg = self.seg_cfg.get("focal_bce", {}) or {}
        self.dice_cfg = self.seg_cfg.get("dice", {}) or {}
        self.tda_cfg = self.cfg.get("temporal_difference_alignment", {}) or {}

    @staticmethod
    def _align_logits_target(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = target.float()
        if target.ndim == 4:
            if logits.ndim == 6:
                logits = logits[:, logits.shape[1] // 2, logits.shape[2] // 2]
            elif logits.ndim == 5:
                logits = logits[:, logits.shape[1] // 2]
        elif target.ndim == 5 and logits.ndim == 6:
            target = target[:, None]
        elif target.ndim == 6 and logits.ndim == 5:
            logits = logits[:, None]
        if logits.shape[-2:] != target.shape[-2:]:
            flat = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
            flat = F.interpolate(flat, size=target.shape[-2:], mode="bilinear", align_corners=False)
            logits = flat.reshape(*logits.shape[:-2], target.shape[-2], target.shape[-1])
        return logits, target

    @staticmethod
    def _flatten_4d(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

    @staticmethod
    def _temporal_view(x: torch.Tensor) -> torch.Tensor | None:
        if x.ndim == 6:
            return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[-3], x.shape[-2], x.shape[-1])
        if x.ndim == 5:
            return x
        return None

    def _focal_bce(self, logits4: torch.Tensor, target4: torch.Tensor) -> torch.Tensor:
        if not bool(self.focal_cfg.get("enabled", True)):
            return logits4.sum() * 0.0
        gamma = float(self.focal_cfg.get("gamma", 2.0))
        alpha = float(self.focal_cfg.get("alpha", 0.75))
        bce = F.binary_cross_entropy_with_logits(logits4, target4, reduction="none")
        prob = torch.sigmoid(logits4)
        pt = prob * target4 + (1.0 - prob) * (1.0 - target4)
        alpha_t = alpha * target4 + (1.0 - alpha) * (1.0 - target4)
        return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()

    def _dice_loss(self, logits4: torch.Tensor, target4: torch.Tensor) -> torch.Tensor:
        if not bool(self.dice_cfg.get("enabled", True)):
            return logits4.sum() * 0.0
        smooth = float(self.dice_cfg.get("smooth", 1.0e-6))
        prob = torch.sigmoid(logits4).flatten(1)
        target = target4.float().flatten(1)
        inter = (prob * target).sum(dim=1)
        denom = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def _seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits4 = self._flatten_4d(logits)
        target4 = self._flatten_4d(target)
        focal_bce = self._focal_bce(logits4, target4)
        dice = self._dice_loss(logits4, target4)
        seg = (
            float(self.focal_cfg.get("weight", 0.5)) * focal_bce
            + float(self.dice_cfg.get("weight", 0.5)) * dice
        )
        return seg, focal_bce, dice

    def _sample_pairs(self, t: int, max_pairs: int, device: torch.device) -> list[tuple[int, int]]:
        pairs = torch.combinations(torch.arange(t, device=device), r=2)
        if pairs.numel() == 0:
            return []
        if max_pairs > 0 and pairs.shape[0] > max_pairs:
            perm = torch.randperm(pairs.shape[0], device=device)[:max_pairs]
            pairs = pairs[perm]
        return [(int(i), int(j)) for i, j in pairs.detach().cpu().tolist()]

    def _tda_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_t = self._temporal_view(logits)
        target_t = self._temporal_view(target)
        if logits_t is None or target_t is None or logits_t.shape[1] < 2:
            return logits.sum() * 0.0

        b, t, c, h, w = logits_t.shape
        pred4 = torch.sigmoid(logits_t.reshape(b * t, c, h, w))
        target4 = target_t.reshape(b * t, c, h, w).float()
        size = int(self.tda_cfg.get("downsample", 128) or 0)
        if size > 0 and pred4.shape[-2:] != (size, size):
            pred4 = F.interpolate(pred4, size=(size, size), mode="bilinear", align_corners=False)
            target4 = F.interpolate(target4, size=(size, size), mode="nearest")

        pred = pred4.reshape(b, t, c, pred4.shape[-2], pred4.shape[-1])
        masks = target4.reshape(b, t, c, target4.shape[-2], target4.shape[-1])
        pairs = self._sample_pairs(t, int(self.tda_cfg.get("max_pairs", 16)), logits.device)
        if not pairs:
            return logits.sum() * 0.0

        total = logits.sum() * 0.0
        eps = float(self.tda_cfg.get("eps", 1.0e-3))
        loss_type = str(self.tda_cfg.get("loss_type", "charbonnier")).lower()
        for i, j in pairs:
            diff = (pred[:, i] - pred[:, j]).abs() - (masks[:, i] - masks[:, j]).abs()
            if loss_type == "l1":
                value = diff.abs().mean()
            else:
                value = torch.sqrt(diff.pow(2) + eps * eps).mean()
            total = total + value
        return total / len(pairs)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del aux
        logits, target = self._align_logits_target(logits, target)
        seg_loss, focal_bce, dice = self._seg_loss(logits, target)
        seg_weight = float(self.seg_cfg.get("weight", 1.0))

        tda_weight = 0.0
        if include_aux and bool(self.tda_cfg.get("enabled", True)):
            tda_weight = _ramp_weight(
                float(self.tda_cfg.get("weight", 0.05)),
                epoch,
                int(self.tda_cfg.get("warmup_epochs", 10)),
            )
            tda_loss = self._tda_loss(logits, target)
        else:
            tda_loss = logits.sum() * 0.0

        total = seg_weight * seg_loss + tda_weight * tda_loss
        items = {
            "loss_total": float(total.detach().cpu()),
            "loss_seg": float(seg_loss.detach().cpu()),
            "loss_focal_bce": float(focal_bce.detach().cpu()),
            "loss_dice": float(dice.detach().cpu()),
            "loss_tda": float(tda_loss.detach().cpu()),
            "tda_weight": float(tda_weight),
            "main_loss": float(total.detach().cpu()),
        }
        return total, items
