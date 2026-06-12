from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_loss import boundary_target
from .segmentation_losses import DiceLoss


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, float(lambd))


class SUMIMinimalityHeads(nn.Module):
    def __init__(self, channels: int = 128, bottleneck_dim: int = 64, num_sources: int = 3):
        super().__init__()
        channels = int(channels)
        bottleneck_dim = int(bottleneck_dim)
        self.mu = nn.Conv2d(channels, bottleneck_dim, 1)
        self.logvar = nn.Conv2d(channels, bottleneck_dim, 1)
        self.source_classifier = nn.Sequential(
            nn.Linear(channels, max(16, channels // 4)),
            nn.GELU(),
            nn.Linear(max(16, channels // 4), int(num_sources)),
        )

    def forward(self, feature: torch.Tensor, *, grl_lambda: float = 0.05) -> dict[str, torch.Tensor]:
        if feature.ndim > 4:
            feature = feature.reshape(-1, feature.shape[-3], feature.shape[-2], feature.shape[-1])
        mu = self.mu(feature)
        logvar = self.logvar(feature).clamp(-8.0, 8.0)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        pooled = feature.mean(dim=(-2, -1))
        source_logits = self.source_classifier(grl(pooled, grl_lambda))
        activation = feature.float().abs().mean(dim=1, keepdim=True)
        return {
            "sumi_ib_kl_tensor": kl,
            "sumi_source_logits": source_logits,
            "sumi_activation32": activation,
        }


class SUMILocalizationLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.dice = DiceLoss()

    @staticmethod
    def _align_pred_target(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim == 4:
            if pred.ndim == 6:
                pred = pred[:, pred.shape[1] // 2, pred.shape[2] // 2]
            elif pred.ndim == 5:
                pred = pred[:, pred.shape[1] // 2]
        elif target.ndim == 5 and pred.ndim == 6:
            target = target[:, None]
        return pred, target

    @staticmethod
    def _resize_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        flat = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1]).float()
        return F.interpolate(flat, size=size, mode="nearest")

    def _mask_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._align_pred_target(pred, target)
        target_rs = self._resize_target(target, pred.shape[-2:])
        pred_4d = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        return self.dice(pred_4d, target_rs) + 0.5 * F.binary_cross_entropy_with_logits(pred_4d, target_rs)

    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._align_pred_target(pred, target)
        target_rs = self._resize_target(target, pred.shape[-2:])
        edge = boundary_target(target_rs)
        pred_4d = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        return F.binary_cross_entropy_with_logits(pred_4d, edge)

    @staticmethod
    def _source_labels(source_names: Any, count: int, device: torch.device) -> torch.Tensor:
        mapping = {"DVI": 0, "CPNET": 1, "OPN": 2}
        if isinstance(source_names, (list, tuple)):
            values = list(source_names)
        else:
            values = [source_names]
        labels = []
        for value in values:
            text = str(value)
            label = 0
            for key, idx in mapping.items():
                if key in text:
                    label = idx
                    break
            labels.append(label)
        if not labels:
            labels = [0]
        while len(labels) < count:
            labels.extend(labels[: count - len(labels)])
        return torch.tensor(labels[:count], device=device, dtype=torch.long)

    def forward(self, aux: dict[str, Any] | None, target: torch.Tensor, *, epoch: int = 0, source_names: Any = None) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.enabled or not aux:
            return target.sum() * 0.0, {}
        total = target.sum() * 0.0
        items: dict[str, float] = {}

        suff_cfg = self.cfg.get("sufficiency", {}) or {}
        suff_start = int(suff_cfg.get("start_epoch", 10))
        if epoch >= suff_start:
            for key, weight in suff_cfg.items():
                if key == "start_epoch" or not isinstance(weight, (int, float)) or float(weight) <= 0:
                    continue
                pred = aux.get(key)
                if pred is None:
                    continue
                if "boundary" in key:
                    value = self._boundary_loss(pred, target)
                else:
                    value = self._mask_loss(pred, target)
                total = total + float(weight) * value
                items[f"sumi_{key}_loss"] = float(value.detach().cpu())
            if items:
                items["sumi_sufficiency_loss"] = sum(v for k, v in items.items() if k.startswith("sumi_") and k.endswith("_loss"))

        min_cfg = self.cfg.get("minimality", {}) or {}
        ib_tensor = aux.get("sumi_ib_kl_tensor")
        ib_weight = float(min_cfg.get("ib_kl_weight", 0.0))
        if ib_tensor is not None:
            total = total + ib_tensor.float() * 0.0
        if ib_tensor is not None and epoch >= int(min_cfg.get("start_epoch", 20)) and ib_weight > 0:
            value = ib_tensor.float()
            total = total + ib_weight * value
            items["sumi_ib_kl"] = float(value.detach().cpu())

        bg_cfg = self.cfg.get("background_suppression", {}) or {}
        activation = aux.get("sumi_activation32")
        if activation is not None and bool(bg_cfg.get("enabled", False)):
            pred, tgt = self._align_pred_target(activation, target)
            tgt_rs = self._resize_target(tgt, pred.shape[-2:])
            pred_4d = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1]).float()
            value = (pred_4d * (1.0 - tgt_rs.float())).mean()
            total = total + float(bg_cfg.get("weight", 0.01)) * value
            items["background_suppression_loss"] = float(value.detach().cpu())

        adv_cfg = self.cfg.get("source_adversarial", {}) or {}
        source_logits = aux.get("sumi_source_logits")
        if source_logits is not None:
            total = total + source_logits.float().sum() * 0.0
        if source_logits is not None and bool(adv_cfg.get("enabled", False)) and epoch >= int(adv_cfg.get("start_epoch", 30)):
            labels = self._source_labels(source_names, source_logits.shape[0], source_logits.device)
            value = F.cross_entropy(source_logits.float(), labels)
            total = total + float(adv_cfg.get("weight", 0.01)) * value
            items["sumi_source_adv_loss"] = float(value.detach().cpu())

        if items:
            items["sumi_loss"] = float(total.detach().cpu())
        return total, items
