from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_entropy(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    prob = prob.float().clamp(eps, 1.0 - eps)
    ent = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
    return ent / 0.69314718056


class CappedAlpha(nn.Module):
    def __init__(self, init: float = 0.001, max_value: float = 0.035):
        super().__init__()
        max_value = max(float(max_value), 1.0e-8)
        ratio = min(max(float(init) / max_value, 1.0e-4), 1.0 - 1.0e-4)
        self.raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.max_value = float(max_value)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) * self.max_value


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        reset, update = gate.chunk(2, dim=1)
        cand = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * cand


class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, min(8, channels)), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalCueUnraveling(nn.Module):
    """Dense TFCU-style temporal cue unraveling.

    Input features are [B,T,C,H,W]. The module does not store raw mask/cue banks;
    it decomposes the current window into momentary, gradual, and cumulative
    anomaly maps, then returns a conservative feature residual.
    """

    def __init__(self, in_dim: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.in_dim = int(in_dim)
        self.hidden_dim = int(cfg.get("hidden_dim", (cfg.get("gradual", {}) or {}).get("hidden_dim", 128)))
        self.detach_history = bool((cfg.get("gradual", {}) or {}).get("detach_history", True))
        self.min_quality = float((cfg.get("gradual", {}) or {}).get("min_quality", 0.25))
        self.cumulative_momentum = float((cfg.get("cumulative", {}) or {}).get("momentum", 0.90))
        self.branch_dropout = float(cfg.get("branch_dropout", 0.10))
        self.alpha = CappedAlpha(float(cfg.get("alpha_init", 0.001)), float(cfg.get("alpha_max", 0.035)))

        self.input_proj = nn.Sequential(
            nn.GroupNorm(32, self.in_dim),
            nn.Conv2d(self.in_dim, self.hidden_dim, 1),
            nn.GELU(),
        )
        self.diff_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 1, self.hidden_dim, 1),
            nn.GELU(),
            DepthwiseBlock(self.hidden_dim),
        )
        self.gru = ConvGRUCell(self.hidden_dim, int((cfg.get("gradual", {}) or {}).get("convgru_kernel", 3)))
        self.grad_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1),
            nn.GELU(),
            DepthwiseBlock(self.hidden_dim),
        )
        self.cum_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, 1),
            nn.GELU(),
            DepthwiseBlock(self.hidden_dim),
        )
        self.gate_head = nn.Conv2d(self.hidden_dim * 3 + self.hidden_dim, 3, 1)
        self.delta_proj = nn.Conv2d(self.hidden_dim, self.in_dim, 1)
        self.logit_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.momentary_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.gradual_head32 = nn.Conv2d(self.hidden_dim, 1, 1)
        self.cumulative_head32 = nn.Conv2d(self.hidden_dim, 1, 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def _quality(self, lowres_logits: torch.Tensor | None, b: int, t: int, h: int, w: int, device, dtype) -> torch.Tensor:
        if lowres_logits is None:
            return torch.ones(b, t, 1, 1, 1, device=device, dtype=dtype)
        prob = torch.sigmoid(lowres_logits.detach().float())
        if prob.shape[-2:] != (h, w):
            flat = prob.reshape(-1, 1, prob.shape[-2], prob.shape[-1])
            prob = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False).reshape(b, t, 1, h, w)
        confidence = 1.0 - binary_entropy(prob).mean(dim=(-2, -1), keepdim=True)
        area = prob.mean(dim=(-2, -1), keepdim=True)
        area_ok = ((area >= 0.002) & (area <= 0.60)).float()
        quality = (confidence * area_ok).clamp(self.min_quality, 1.0)
        return quality.to(device=device, dtype=dtype)

    def forward(
        self,
        features: torch.Tensor,
        lowres_logits: torch.Tensor | None = None,
        video_ids: list[str] | None = None,
        frame_indices: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,T,C,H,W], got {tuple(features.shape)}")
        b, t, _c, h, w = features.shape
        feat = self.input_proj(features.reshape(b * t, self.in_dim, h, w)).reshape(b, t, self.hidden_dim, h, w)

        mom = torch.zeros_like(feat)
        if t > 1:
            f0 = F.normalize(feat[:, 1:].float(), dim=2).to(dtype=feat.dtype)
            f1 = F.normalize(feat[:, :-1].float(), dim=2).to(dtype=feat.dtype)
            diff = torch.abs(feat[:, 1:] - feat[:, :-1])
            cos = 1.0 - (f0 * f1).sum(dim=2, keepdim=True)
            x = torch.cat([diff.reshape(b * (t - 1), self.hidden_dim, h, w), cos.reshape(b * (t - 1), 1, h, w)], dim=1)
            mom[:, 1:] = self.diff_proj(x).reshape(b, t - 1, self.hidden_dim, h, w)
            mom[:, 0] = mom[:, 1]
        else:
            mom[:, 0] = self.diff_proj(torch.cat([feat[:, 0], torch.zeros(b, 1, h, w, device=feat.device, dtype=feat.dtype)], dim=1))

        quality = self._quality(lowres_logits, b, t, h, w, feat.device, feat.dtype)
        h_state = torch.zeros(b, self.hidden_dim, h, w, device=feat.device, dtype=feat.dtype)
        gradual = []
        for idx in range(t):
            new_state = self.gru(mom[:, idx], h_state)
            q = quality[:, idx]
            if self.detach_history:
                h_state = q * new_state + (1.0 - q) * h_state.detach()
            else:
                h_state = q * new_state + (1.0 - q) * h_state
            gradual.append(self.grad_head(torch.cat([mom[:, idx], h_state], dim=1)))
        gradual_t = torch.stack(gradual, dim=1)

        fwd = []
        state = torch.zeros_like(gradual_t[:, 0])
        for idx in range(t):
            state = self.cumulative_momentum * state + (1.0 - self.cumulative_momentum) * gradual_t[:, idx]
            fwd.append(state)
        bwd = []
        state = torch.zeros_like(gradual_t[:, -1])
        for idx in range(t - 1, -1, -1):
            state = self.cumulative_momentum * state + (1.0 - self.cumulative_momentum) * gradual_t[:, idx]
            bwd.append(state)
        bwd = list(reversed(bwd))
        fwd_t = torch.stack(fwd, dim=1)
        bwd_t = torch.stack(bwd, dim=1)
        cum_in = torch.cat([fwd_t, bwd_t, gradual_t], dim=2).reshape(b * t, self.hidden_dim * 3, h, w)
        cumulative = self.cum_head(cum_in).reshape(b, t, self.hidden_dim, h, w)

        gate_in = torch.cat([mom, gradual_t, cumulative, feat], dim=2).reshape(b * t, self.hidden_dim * 4, h, w)
        gate_logits = self.gate_head(gate_in).reshape(b, t, 3, h, w)
        if self.training and self.branch_dropout > 0.0:
            keep = torch.rand(b, t, 3, 1, 1, device=features.device, dtype=features.dtype) >= self.branch_dropout
            gate_logits = gate_logits.masked_fill(~keep, -20.0)
        gate = torch.softmax(gate_logits.float(), dim=2).to(dtype=features.dtype)
        cue = gate[:, :, 0:1] * mom + gate[:, :, 1:2] * gradual_t + gate[:, :, 2:3] * cumulative
        alpha = self.alpha().to(dtype=features.dtype)
        feat_delta = alpha * self.delta_proj(cue.reshape(b * t, self.hidden_dim, h, w)).reshape(b, t, self.in_dim, h, w)
        logit_delta32 = self.logit_head(cue.reshape(b * t, self.hidden_dim, h, w)).reshape(b, t, 1, h, w)
        momentary_map32 = self.momentary_head(mom.reshape(b * t, self.hidden_dim, h, w)).reshape(b, t, 1, h, w)
        gradual_map32 = self.gradual_head32(gradual_t.reshape(b * t, self.hidden_dim, h, w)).reshape(b, t, 1, h, w)
        cumulative_map32 = self.cumulative_head32(cumulative.reshape(b * t, self.hidden_dim, h, w)).reshape(b, t, 1, h, w)
        debug = {
            "tcu_alpha": float(self.alpha().detach().cpu()),
            "tcu_gate_momentary_mean": float(gate[:, :, 0].detach().float().mean().cpu()),
            "tcu_gate_gradual_mean": float(gate[:, :, 1].detach().float().mean().cpu()),
            "tcu_gate_cumulative_mean": float(gate[:, :, 2].detach().float().mean().cpu()),
            "tcu_quality_mean": float(quality.detach().float().mean().cpu()),
        }
        return {
            "feat_delta": feat_delta,
            "logit_delta32": logit_delta32,
            "momentary_map32": momentary_map32,
            "gradual_map32": gradual_map32,
            "cumulative_map32": cumulative_map32,
            "tcu_momentary_mask32": momentary_map32,
            "tcu_gradual_mask32": gradual_map32,
            "tcu_cumulative_mask32": cumulative_map32,
            "gate": gate,
            "quality": quality,
            "debug": debug,
        }
