from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .videomt_loss import dice_loss_from_logits, mask_to_edge


def _binary_components(mask: torch.Tensor, max_components: int, min_area: int) -> list[torch.Tensor]:
    """Connected components on a 2D CPU bool mask. Returns bool masks."""
    mask_np = mask.detach().to(device="cpu", dtype=torch.bool)
    h, w = int(mask_np.shape[0]), int(mask_np.shape[1])
    visited = torch.zeros((h, w), dtype=torch.bool)
    comps: list[tuple[int, torch.Tensor]] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask_np[y, x]:
                continue
            comp = torch.zeros((h, w), dtype=torch.bool)
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            area = 0
            while q:
                cy, cx = q.popleft()
                comp[cy, cx] = True
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and bool(mask_np[ny, nx]):
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if area >= min_area:
                comps.append((area, comp))
    comps.sort(key=lambda item: item[0], reverse=True)
    return [comp for _area, comp in comps[:max_components]]


def _dice_assign(query_probs: torch.Tensor, components: torch.Tensor) -> list[tuple[int, int]]:
    """Min-cost Dice assignment; uses Hungarian when scipy is available."""
    if components.numel() == 0:
        return []
    q = query_probs.shape[0]
    c = components.shape[0]
    qp = query_probs.reshape(q, -1)
    ct = components.reshape(c, -1)
    inter = torch.einsum("qn,cn->qc", qp, ct)
    denom = qp.sum(dim=1, keepdim=True) + ct.sum(dim=1).unsqueeze(0)
    dice = (2.0 * inter + 1.0e-6) / (denom + 1.0e-6)
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment((1.0 - dice).detach().cpu().numpy())
        return [(int(qi), int(ci)) for qi, ci in zip(row_ind.tolist(), col_ind.tolist())]
    except Exception:
        pass

    pairs: list[tuple[float, int, int]] = []
    for qi in range(q):
        for ci in range(c):
            pairs.append((float(dice[qi, ci].detach().cpu()), qi, ci))
    pairs.sort(reverse=True)
    used_q: set[int] = set()
    used_c: set[int] = set()
    out: list[tuple[int, int]] = []
    for _score, qi, ci in pairs:
        if qi in used_q or ci in used_c:
            continue
        used_q.add(qi)
        used_c.add(ci)
        out.append((qi, ci))
        if len(used_c) >= c:
            break
    return out


class VideoMTQueryMaskLoss(nn.Module):
    """Final VidEoMT loss: final mask supervision plus query-level mask supervision."""

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

        qcfg = cfg.get("query_mask", {}) or {}
        self.query_enabled = bool(qcfg.get("enabled", True))
        self.query_bce_weight = float(qcfg.get("bce_weight", 0.5))
        self.query_dice_weight = float(qcfg.get("dice_weight", 0.5))
        mcfg = qcfg.get("matching", {}) or {}
        self.max_components = int(mcfg.get("max_components", 32))
        self.min_component_area = int(mcfg.get("min_component_area", 16))
        ncfg = qcfg.get("no_object", {}) or {}
        self.no_object_enabled = bool(ncfg.get("enabled", True))
        self.no_object_weight = float(ncfg.get("weight", 0.1))

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
            masks = masks.reshape(*logits.shape)
        return logits, masks

    @staticmethod
    def _flatten_2d(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

    def _edge_loss(self, aux: dict[str, Any], masks_2d: torch.Tensor, logits_2d: torch.Tensor) -> torch.Tensor:
        edge_logits = aux.get("edge_logits") if isinstance(aux, dict) else None
        if edge_logits is None and isinstance(aux, dict):
            edge_logits = aux.get("boundary128")
        if self.edge_weight <= 0 or edge_logits is None or not torch.is_tensor(edge_logits):
            return logits_2d.sum() * 0.0
        edge_logits_2d = self._flatten_2d(edge_logits)
        edge_target = F.interpolate(masks_2d, size=edge_logits_2d.shape[-2:], mode="nearest")
        edge_target = mask_to_edge(edge_target, kernel_size=self.edge_kernel_size)
        return F.binary_cross_entropy_with_logits(edge_logits_2d, edge_target)

    def _query_targets(self, query_logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qlog = query_logits
        if qlog.ndim == 5:
            qlog = qlog[:, None]
        if masks.ndim == 5:
            masks = masks[:, None]
        b, w, t, q, hq, wq = qlog.shape
        mask_small = F.interpolate(
            masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1]).float(),
            size=(hq, wq),
            mode="nearest",
        ).reshape(b, w, t, 1, hq, wq)
        query_targets = torch.zeros((b, w, t, q, hq, wq), device=qlog.device, dtype=qlog.dtype)
        query_valid = torch.zeros((b, w, t, q), device=qlog.device, dtype=torch.bool)
        probs = torch.sigmoid(qlog.detach())
        for bi in range(b):
            for wi in range(w):
                for ti in range(t):
                    comps = _binary_components(
                        mask_small[bi, wi, ti, 0] > 0.5,
                        max_components=min(self.max_components, q),
                        min_area=self.min_component_area,
                    )
                    if not comps:
                        continue
                    comp_tensor = torch.stack(comps, dim=0).to(device=qlog.device, dtype=qlog.dtype)
                    pairs = _dice_assign(probs[bi, wi, ti], comp_tensor)
                    for qi, ci in pairs:
                        query_targets[bi, wi, ti, qi] = comp_tensor[ci]
                        query_valid[bi, wi, ti, qi] = True
        return query_targets, query_valid

    def _query_loss(self, aux: dict[str, Any], masks: torch.Tensor, zero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_logits = aux.get("query_logits") if isinstance(aux, dict) else None
        if not self.query_enabled or query_logits is None or not torch.is_tensor(query_logits):
            return zero, zero, zero
        if query_logits.ndim == 4:
            query_logits = query_logits[:, None, None]
            masks = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1])[:, None, None]
        query_targets, query_valid = self._query_targets(query_logits, masks)
        qlog = query_logits if query_logits.ndim == 6 else query_logits[:, None]
        matched = query_valid
        unmatched = ~matched
        if matched.any():
            loss_q_bce = F.binary_cross_entropy_with_logits(qlog[matched], query_targets[matched])
            loss_q_dice = dice_loss_from_logits(qlog[matched].unsqueeze(1), query_targets[matched].unsqueeze(1), eps=self.dice_eps)
        else:
            loss_q_bce = zero
            loss_q_dice = zero
        if self.no_object_enabled and unmatched.any():
            loss_no_obj = F.binary_cross_entropy_with_logits(qlog[unmatched], torch.zeros_like(qlog[unmatched]))
        else:
            loss_no_obj = zero
        return loss_q_bce, loss_q_dice, loss_no_obj

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
        zero = logits_2d.sum() * 0.0
        loss_edge = self._edge_loss(aux, masks_2d, logits_2d) if include_aux else zero
        if include_aux:
            loss_q_bce, loss_q_dice, loss_q_no_obj = self._query_loss(aux, masks, zero)
        else:
            loss_q_bce, loss_q_dice, loss_q_no_obj = zero, zero, zero
        total = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.edge_weight * loss_edge
            + self.query_bce_weight * loss_q_bce
            + self.query_dice_weight * loss_q_dice
            + self.no_object_weight * loss_q_no_obj
        )
        items = {
            "loss_total": float(total.detach().cpu()),
            "loss_bce": float(loss_bce.detach().cpu()),
            "loss_dice": float(loss_dice.detach().cpu()),
            "loss_query_bce": float(loss_q_bce.detach().cpu()),
            "loss_query_dice": float(loss_q_dice.detach().cpu()),
            "loss_query_no_object": float(loss_q_no_obj.detach().cpu()),
            "loss_edge": float(loss_edge.detach().cpu()),
            "main_loss": float(total.detach().cpu()),
        }
        return total, items
