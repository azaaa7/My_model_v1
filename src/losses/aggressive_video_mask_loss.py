from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_iml_original_loss import _flatten_to_nchw, make_edge_mask


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob.reshape(prob.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class AggressiveVideoMaskLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        final_cfg = cfg.get("final_mask", {}) or {}
        aux_cfg = cfg.get("aux_mask", {}) or {}
        edge_cfg = cfg.get("edge", {}) or {}
        temp_cfg = cfg.get("temporal_consistency", {}) or {}
        self.final_bce_weight = float(final_cfg.get("bce_weight", 1.0))
        self.final_dice_weight = float(final_cfg.get("dice_weight", 1.0))
        self.aux_enabled = bool(aux_cfg.get("enabled", True))
        self.aux_weight = float(aux_cfg.get("weight", 0.40))
        self.aux_bce_weight = float(aux_cfg.get("bce_weight", 1.0))
        self.aux_dice_weight = float(aux_cfg.get("dice_weight", 1.0))
        self.edge_enabled = bool(edge_cfg.get("enabled", True))
        self.edge_weight = float(edge_cfg.get("weight", 5.0))
        self.edge_width = int(edge_cfg.get("width", 7))
        self.temporal_enabled = bool(temp_cfg.get("enabled", False))
        self.temporal_weight = float(temp_cfg.get("weight", 0.01))
        self.temporal_gamma = float(temp_cfg.get("gamma", 5.0))
        self.temporal_warmup_epochs = int(temp_cfg.get("warmup_epochs", 30))

    def _mask_loss(self, logits: torch.Tensor, target: torch.Tensor, bce_weight: float, dice_weight: float) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = _soft_dice_loss(logits, target)
        return bce * bce_weight + dice * dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        aux = aux or {}
        pred = _flatten_to_nchw(logits)
        gt = _flatten_to_nchw(target).float()
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")

        loss_final = self._mask_loss(pred, gt, self.final_bce_weight, self.final_dice_weight)
        total = loss_final
        items = {"loss_final_mask": float(loss_final.detach().cpu())}

        if self.aux_enabled and include_aux:
            aux_list = aux.get("aux_logits", []) or []
            if aux_list:
                aux_loss = pred.sum() * 0.0
                for aux_logits in aux_list:
                    aux_pred = _flatten_to_nchw(aux_logits)
                    aux_gt = gt
                    if aux_gt.shape[-2:] != aux_pred.shape[-2:]:
                        aux_gt = F.interpolate(aux_gt, size=aux_pred.shape[-2:], mode="nearest")
                    aux_loss = aux_loss + self._mask_loss(aux_pred, aux_gt, self.aux_bce_weight, self.aux_dice_weight)
                aux_loss = aux_loss / len(aux_list)
                total = total + self.aux_weight * aux_loss
                items["loss_aux_mask"] = float(aux_loss.detach().cpu())
            else:
                items["loss_aux_mask"] = 0.0
        else:
            items["loss_aux_mask"] = 0.0

        if self.edge_enabled:
            edge = make_edge_mask(gt, width=self.edge_width)
            bce_map = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
            loss_edge = (bce_map * edge).sum() / edge.sum().clamp_min(1.0)
            total = total + self.edge_weight * loss_edge
            items["loss_edge"] = float(loss_edge.detach().cpu())
        else:
            items["loss_edge"] = 0.0

        loss_temp = pred.sum() * 0.0
        if self.temporal_enabled and include_aux and epoch is not None and epoch >= self.temporal_warmup_epochs:
            features = aux.get("features32")
            if features is not None and logits.ndim == 6:
                prob = torch.sigmoid(logits)
                feat = features
                if feat.shape[2] > 1:
                    feat_n = F.normalize(feat.float(), dim=2).to(dtype=feat.dtype)
                    feat_diff = (feat_n[:, :, 1:] - feat_n[:, :, :-1]).abs().mean(dim=2, keepdim=True)
                    feat_diff = F.interpolate(
                        feat_diff.reshape(-1, 1, feat_diff.shape[-2], feat_diff.shape[-1]),
                        size=prob.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(prob.shape[0], prob.shape[1], prob.shape[2] - 1, 1, prob.shape[-2], prob.shape[-1])
                    temp_weight = torch.exp(-self.temporal_gamma * feat_diff)
                    prob_diff = (prob[:, :, 1:] - prob[:, :, :-1]).abs()
                    loss_temp = (temp_weight * prob_diff).mean()
                    total = total + self.temporal_weight * loss_temp
        items["loss_temp"] = float(loss_temp.detach().cpu())
        items["main_loss"] = float(total.detach().cpu())
        return total, items
