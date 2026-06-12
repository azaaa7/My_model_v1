from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def ramp_weight(base_weight: float, epoch: int | None, warmup_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if epoch is None or warmup_epochs <= 0:
        return float(base_weight)
    ratio = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return float(base_weight) * ratio


def _odd_kernel(value: int) -> int:
    value = max(int(value), 1)
    return value if value % 2 == 1 else value + 1


class CompositeForensicLoss(nn.Module):
    """Region + boundary + temporal + deep supervision loss for video forensics."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self.region_cfg = self.cfg.get("region", {}) or {}
        self.boundary_cfg = self.cfg.get("boundary", {}) or {}
        self.temporal_cfg = self.cfg.get("pairwise_temporal", {}) or {}
        self.deep_cfg = self.cfg.get("deep_supervision", {}) or {}
        self.token_cfg = self.cfg.get("token_separation", {}) or {}
        self.ttf_reg_cfg = self.cfg.get("ttf_regularization", {}) or {}
        self.eps = float(self.cfg.get("eps", 1.0e-6))

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
    def _align_aux_pred_target(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = target.float()
        if target.ndim == 4:
            if pred.ndim == 6:
                pred = pred[:, pred.shape[1] // 2, pred.shape[2] // 2]
            elif pred.ndim == 5:
                pred = pred[:, pred.shape[1] // 2]
        elif target.ndim == 5 and pred.ndim == 6:
            target = target[:, None]
        elif target.ndim == 6 and pred.ndim == 5:
            pred = pred[:, None]
        pred4 = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        target4 = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1])
        if pred4.shape[0] != target4.shape[0]:
            raise RuntimeError(f"Aux head batch mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if target4.shape[-2:] != pred4.shape[-2:]:
            target4 = F.interpolate(target4, size=pred4.shape[-2:], mode="nearest")
        return pred4, target4

    def _focal_bce(self, logits4: torch.Tensor, target4: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        gamma = float(cfg.get("gamma", 2.0))
        alpha = float(cfg.get("alpha", 0.75))
        pos_weight = None
        if bool(cfg.get("adaptive_pos_weight", False)):
            pos = target4.sum().detach()
            neg = target4.numel() - pos
            low, high = cfg.get("pos_weight_clip", [1.0, 20.0])
            pos_weight = (neg / pos.clamp_min(self.eps)).clamp(float(low), float(high)).to(logits4.device)
        bce = F.binary_cross_entropy_with_logits(logits4, target4, pos_weight=pos_weight, reduction="none")
        probs = torch.sigmoid(logits4)
        pt = probs * target4 + (1.0 - probs) * (1.0 - target4)
        alpha_t = alpha * target4 + (1.0 - alpha) * (1.0 - target4)
        return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()

    def _dice_loss(self, logits4: torch.Tensor, target4: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits4).flatten(1)
        target = target4.float().flatten(1)
        inter = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def _focal_tversky(self, logits4: torch.Tensor, target4: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        probs = torch.sigmoid(logits4).flatten(1)
        target = target4.float().flatten(1)
        alpha = float(cfg.get("alpha", 0.3))
        beta = float(cfg.get("beta", 0.7))
        gamma = float(cfg.get("gamma", 0.75))
        smooth = float(cfg.get("smooth", self.eps))
        tp = (probs * target).sum(dim=1)
        fp = (probs * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - probs) * target).sum(dim=1)
        tv = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return (1.0 - tv).pow(gamma).mean()

    def _region_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits4 = self._flatten_4d(logits)
        target4 = self._flatten_4d(target)
        bce_cfg = self.region_cfg.get("focal_bce", {}) or {}
        tv_cfg = self.region_cfg.get("focal_tversky", {}) or {}
        bce = self._focal_bce(logits4, target4, bce_cfg)
        tv = self._focal_tversky(logits4, target4, tv_cfg)
        return float(bce_cfg.get("weight", 0.6)) * bce + float(tv_cfg.get("weight", 0.4)) * tv

    def _boundary_edge_band(
        self,
        target4: torch.Tensor,
        kernel_size: int,
        band_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_size = _odd_kernel(kernel_size)
        band_width = _odd_kernel(band_width)
        pad = kernel_size // 2
        target4 = target4.float().clamp(0.0, 1.0)
        dilated = F.max_pool2d(target4, kernel_size=kernel_size, stride=1, padding=pad)
        eroded = 1.0 - F.max_pool2d(1.0 - target4, kernel_size=kernel_size, stride=1, padding=pad)
        edge = (dilated - eroded).clamp(0.0, 1.0)
        band = F.max_pool2d(edge, kernel_size=band_width, stride=1, padding=band_width // 2).clamp(0.0, 1.0)
        return edge, band

    def _boundary_loss(self, logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any] | None = None) -> torch.Tensor:
        cfg = cfg or self.boundary_cfg
        logits4 = self._flatten_4d(logits)
        target4 = self._flatten_4d(target)
        edge, band = self._boundary_edge_band(
            target4,
            kernel_size=int(cfg.get("kernel_size", 5)),
            band_width=int(cfg.get("band_width", 5)),
        )
        band_sum = band.sum()
        if float(band_sum.detach().cpu()) <= 0.0:
            return logits4.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(logits4, edge, reduction="none")
        bce = (bce * band).sum() / band_sum.clamp_min(self.eps)
        probs = torch.sigmoid(logits4) * band
        edge = edge * band
        inter = (probs * edge).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + edge.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + self.eps) / (denom + self.eps)).mean()
        return float(cfg.get("bce_weight", 0.7)) * bce + float(cfg.get("dice_weight", 0.3)) * dice

    @staticmethod
    def _temporal_view(x: torch.Tensor) -> torch.Tensor | None:
        if x.ndim == 6:
            return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[-3], x.shape[-2], x.shape[-1])
        if x.ndim == 5:
            return x
        return None

    def _pairwise_temporal_loss(self, logits: torch.Tensor, target: torch.Tensor, epoch: int | None) -> torch.Tensor:
        logits_t = self._temporal_view(logits)
        target_t = self._temporal_view(target)
        if logits_t is None or target_t is None or logits_t.shape[1] <= 1:
            return logits.sum() * 0.0
        b, t, c, h, w = logits_t.shape
        size = int(self.temporal_cfg.get("downsample", 128) or 0)
        logits4 = logits_t.reshape(b * t, c, h, w)
        target4 = target_t.reshape(b * t, c, h, w).float()
        if size > 0 and (h, w) != (size, size):
            logits4 = F.interpolate(logits4, size=(size, size), mode="bilinear", align_corners=False)
            target4 = F.interpolate(target4, size=(size, size), mode="nearest")
        probs = torch.sigmoid(logits4).reshape(b, t, c, logits4.shape[-2], logits4.shape[-1])
        masks = target4.reshape(b, t, c, target4.shape[-2], target4.shape[-1])
        pairs = [(i, j) for i in range(t) for j in range(i + 1, t)]
        max_pairs = int(self.temporal_cfg.get("max_pairs", 0) or 0)
        if max_pairs > 0 and len(pairs) > max_pairs:
            ids = torch.linspace(0, len(pairs) - 1, steps=max_pairs).round().long().tolist()
            pairs = [pairs[i] for i in ids]
        total = logits.sum() * 0.0
        pos_w = float(self.temporal_cfg.get("positive_delta_weight", 1.0))
        neg_w = float(self.temporal_cfg.get("negative_delta_weight", 0.35))
        charbonnier_eps = float(self.temporal_cfg.get("charbonnier_eps", 1.0e-3))
        for i, j in pairs:
            pred_delta = (probs[:, i] - probs[:, j]).abs()
            gt_delta = (masks[:, i] - masks[:, j]).abs()
            diff = pred_delta - gt_delta
            if str(self.temporal_cfg.get("loss_type", "charbonnier")).lower() == "l1":
                rho = diff.abs()
            else:
                rho = torch.sqrt(diff.pow(2) + charbonnier_eps**2)
            weights = torch.where(gt_delta > 1.0e-4, torch.full_like(gt_delta, pos_w), torch.full_like(gt_delta, neg_w))
            total = total + (rho * weights).mean()
        return total / max(len(pairs), 1)

    def _mask_head_loss(self, pred4: torch.Tensor, target4: torch.Tensor) -> torch.Tensor:
        focal_cfg = {"gamma": 2.0, "alpha": 0.75, "adaptive_pos_weight": False}
        return self._focal_bce(pred4, target4, focal_cfg) + self._dice_loss(pred4, target4)

    def _deep_supervision_loss(self, aux: dict[str, Any] | None, target: torch.Tensor, items: dict[str, float]) -> torch.Tensor:
        if not aux:
            return target.sum() * 0.0
        heads = self.deep_cfg.get("heads", {}) or {}
        total = target.sum() * 0.0
        active_weight = 0.0
        for name, head_cfg in heads.items():
            head_cfg = head_cfg or {}
            pred = aux.get(name)
            weight = float(head_cfg.get("weight", 1.0))
            if not bool(head_cfg.get("enabled", False)) or pred is None or weight <= 0 or not torch.is_tensor(pred):
                continue
            pred4, target4 = self._align_aux_pred_target(pred, target)
            if "boundary" in str(name):
                value = self._boundary_loss(pred4, target4, self.boundary_cfg)
            else:
                value = self._mask_head_loss(pred4, target4)
            total = total + weight * value
            active_weight += weight
            items[f"deep_{name}_loss"] = float(value.detach().cpu())
        if active_weight <= 0:
            return target.sum() * 0.0
        if bool(self.deep_cfg.get("normalize_by_active_heads", True)):
            total = total / active_weight
        return total

    def _token_separation_loss(self, aux: dict[str, Any] | None, target: torch.Tensor) -> torch.Tensor:
        if not aux or not torch.is_tensor(aux.get("ttf_tokens")):
            return target.sum() * 0.0
        tokens = aux["ttf_tokens"]
        if tokens.ndim != 5:
            return target.sum() * 0.0
        b, m, k, n, c = tokens.shape
        side = int(math.sqrt(n))
        if side * side != n or target.ndim != 6:
            return tokens.sum() * 0.0
        target4 = target.reshape(b * m * k, target.shape[-3], target.shape[-2], target.shape[-1]).float()
        gt = F.interpolate(target4, size=(side, side), mode="nearest").reshape(b, m, k, n)
        tok = F.normalize(tokens.float(), dim=-1)
        fg = gt > 0.5
        bg = gt < 0.1
        min_pixels = int(self.token_cfg.get("min_pixels_per_class", 8))
        if int(fg.sum().detach().cpu()) < min_pixels or int(bg.sum().detach().cpu()) < min_pixels:
            return tokens.sum() * 0.0
        fg_proto = F.normalize(tok[fg].mean(dim=0, keepdim=True), dim=-1)
        bg_proto = F.normalize(tok[bg].mean(dim=0, keepdim=True), dim=-1)
        margin = float(self.token_cfg.get("margin", 0.2))
        return F.relu((fg_proto * bg_proto).sum(dim=-1) + margin).mean()

    def _ttf_regularization_loss(self, aux: dict[str, Any] | None, target: torch.Tensor) -> torch.Tensor:
        if not aux or not torch.is_tensor(aux.get("ttf_residual_energy")):
            return target.sum() * 0.0
        energy = aux["ttf_residual_energy"].float()
        if energy.ndim != 4 or target.ndim != 6:
            return energy.sum() * 0.0
        b, m, k, n = energy.shape
        side = int(math.sqrt(n))
        if side * side != n or tuple(target.shape[:3]) != (b, m, k):
            return energy.sum() * 0.0
        target4 = target.reshape(b * m * k, target.shape[-3], target.shape[-2], target.shape[-1]).float()
        gt = F.interpolate(target4, size=(side, side), mode="nearest").reshape(b, m, k, n)
        background = (gt < 0.1).float()
        denom = background.sum().clamp_min(1.0)
        return (energy * background).sum() / denom

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits, target = self._align_logits_target(logits, target)
        total = logits.sum() * 0.0
        items: dict[str, float] = {}

        if bool(self.region_cfg.get("enabled", True)) and float(self.region_cfg.get("weight", 1.0)) > 0:
            value = self._region_loss(logits, target)
            total = total + float(self.region_cfg.get("weight", 1.0)) * value
            items["region_loss"] = float(value.detach().cpu())

        if bool(self.boundary_cfg.get("enabled", True)) and float(self.boundary_cfg.get("weight", 0.0)) > 0:
            value = self._boundary_loss(logits, target)
            total = total + float(self.boundary_cfg.get("weight", 0.0)) * value
            items["boundary_loss"] = float(value.detach().cpu())

        if include_aux and bool(self.temporal_cfg.get("enabled", False)):
            weight = ramp_weight(
                float(self.temporal_cfg.get("weight", 0.0)),
                epoch,
                int(self.temporal_cfg.get("warmup_epochs", 0)),
            )
            if weight > 0:
                value = self._pairwise_temporal_loss(logits, target, epoch)
                total = total + weight * value
                items["pairwise_temporal_loss"] = float(value.detach().cpu())

        if include_aux and bool(self.deep_cfg.get("enabled", False)):
            weight = float(self.deep_cfg.get("weight", 0.0))
            if weight > 0:
                value = self._deep_supervision_loss(aux, target, items)
                total = total + weight * value
                items["deep_supervision_loss"] = float(value.detach().cpu())

        if include_aux and bool(self.token_cfg.get("enabled", False)):
            weight = ramp_weight(
                float(self.token_cfg.get("weight", 0.0)),
                epoch,
                int(self.token_cfg.get("warmup_epochs", 0)),
            )
            if weight > 0:
                value = self._token_separation_loss(aux, target)
                total = total + weight * value
                items["token_separation_loss"] = float(value.detach().cpu())

        if include_aux and bool(self.ttf_reg_cfg.get("enabled", False)):
            weight = ramp_weight(
                float(self.ttf_reg_cfg.get("weight", 0.0)),
                epoch,
                int(self.ttf_reg_cfg.get("warmup_epochs", 0)),
            )
            if weight > 0:
                value = self._ttf_regularization_loss(aux, target)
                total = total + weight * value
                items["ttf_regularization_loss"] = float(value.detach().cpu())

        items["main_loss"] = float(total.detach().cpu())
        return total, items
