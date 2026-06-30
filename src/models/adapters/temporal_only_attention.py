from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_prob)
        return x * keep / keep_prob


class TemporalOnlyAttention(nn.Module):
    def __init__(self, channels: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.channels = int(channels)
        self.dim = int(cfg.get("dim", 64))
        self.heads = int(cfg.get("heads", 4))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.use_temporal_pos = bool(cfg.get("use_temporal_pos", True))
        self.alpha_max = float(cfg.get("alpha_max", 0.03))
        self.drop_path = _DropPath(float(cfg.get("drop_path_prob", 0.10)))
        self.max_frames = int(cfg.get("max_frames", 8))

        self.down = nn.Conv2d(self.channels, self.dim, kernel_size=1, bias=False)
        self.attn = nn.MultiheadAttention(self.dim, self.heads, dropout=self.dropout, batch_first=True)
        self.up = nn.Conv2d(self.dim, self.channels, kernel_size=1, bias=True)
        if bool(cfg.get("zero_init_out", True)):
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_frames, self.dim))
        alpha_init = float(cfg.get("alpha_init", 0.0))
        if self.alpha_max > 0:
            alpha_init = max(0.0, min(alpha_init, self.alpha_max))
            init_ratio = alpha_init / self.alpha_max if self.alpha_max > 0 else 0.0
            init_ratio = min(max(init_ratio, 1.0e-6), 1.0 - 1.0e-6)
            self.alpha_raw = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_ratio)).item(), dtype=torch.float32))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return x, {"temporal_only_attn_enabled": False}
        if x.ndim != 5:
            raise ValueError(f"TemporalOnlyAttention expects [G,K,C,H,W], got {tuple(x.shape)}")

        g, k, c, h, w = x.shape
        z = self.down(x.reshape(g * k, c, h, w)).reshape(g, k, self.dim, h, w)
        tokens = z.permute(0, 3, 4, 1, 2).reshape(g * h * w, k, self.dim)
        if self.use_temporal_pos:
            tokens = tokens + self.temporal_pos[:, :k]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = attn_out.reshape(g, h, w, k, self.dim).permute(0, 3, 4, 1, 2).reshape(g * k, self.dim, h, w)
        res = self.up(attn_out).reshape(g, k, c, h, w)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_raw) if self.alpha_max > 0 else 0.0
        out = x + self.drop_path(res) * alpha
        debug = {
            "temporal_only_attn_enabled": True,
            "temporal_only_attn_alpha": float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha),
            "temporal_only_attn_res_norm": float(res.detach().float().norm().cpu()),
        }
        return out, debug
