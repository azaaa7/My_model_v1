from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 1, 3, 4, 2)
        x_norm = self.norm(x_perm)
        return x_norm.permute(0, 1, 4, 2, 3).contiguous()


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


class TemporalDifferenceGatedExcitation(nn.Module):
    def __init__(self, channels: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.channels = int(channels)
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.use_adjacent_diff = bool(cfg.get("use_adjacent_diff", True))
        self.use_stride2_diff = bool(cfg.get("use_stride2_diff", True))
        self.zero_init_value = bool(cfg.get("zero_init_value", True))
        self.beta_max = float(cfg.get("beta_max", 0.03))
        self.drop_path = _DropPath(float(cfg.get("drop_path_prob", 0.05)))

        self.norm = _ChannelLayerNorm(self.channels)
        self.down = nn.Conv2d(self.channels, self.hidden_dim, kernel_size=1, bias=False)

        diff_in = self.hidden_dim * int(self.use_adjacent_diff) + self.hidden_dim * int(self.use_stride2_diff)
        if diff_in <= 0:
            diff_in = self.hidden_dim
        groups = min(32, self.hidden_dim)
        while groups > 1 and self.hidden_dim % groups != 0:
            groups -= 1
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(diff_in, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True),
        )

        gate_hidden = max(8, self.hidden_dim // 2)
        self.channel_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.channels),
        )
        self.spatial_gate = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1, bias=True)
        nn.init.constant_(self.spatial_gate.bias, float(cfg.get("gate_bias_init", -2.0)))
        nn.init.constant_(self.channel_gate[-1].bias, float(cfg.get("gate_bias_init", -2.0)))

        self.value_proj = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1, bias=True)
        if self.zero_init_value:
            nn.init.zeros_(self.value_proj.weight)
            nn.init.zeros_(self.value_proj.bias)

        beta_init = float(cfg.get("beta_init", 0.0))
        if self.beta_max > 0:
            beta_init = max(0.0, min(beta_init, self.beta_max))
            init_ratio = beta_init / self.beta_max if self.beta_max > 0 else 0.0
            init_ratio = min(max(init_ratio, 1.0e-6), 1.0 - 1.0e-6)
            self.beta_raw = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_ratio)).item(), dtype=torch.float32))
        else:
            self.beta_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return x, {"tdgx_enabled": False}
        if x.ndim != 5:
            raise ValueError(f"TDGX expects [G,K,C,H,W], got {tuple(x.shape)}")

        g, k, c, h, w = x.shape
        x_norm = self.norm(x)
        z = self.down(x_norm.reshape(g * k, c, h, w)).reshape(g, k, self.hidden_dim, h, w)

        diffs = []
        if self.use_adjacent_diff:
            d1 = torch.zeros_like(z)
            d1[:, 1:] = torch.abs(z[:, 1:] - z[:, :-1])
            diffs.append(d1)
        if self.use_stride2_diff:
            d2 = torch.zeros_like(z)
            d2[:, 2:] = torch.abs(z[:, 2:] - z[:, :-2])
            diffs.append(d2)
        if not diffs:
            diffs = [torch.zeros_like(z)]
        diff = torch.cat(diffs, dim=2).reshape(g * k, -1, h, w)
        m = self.diff_encoder(diff)

        channel = self.channel_gate(m.mean(dim=(2, 3))).reshape(g * k, c, 1, 1)
        spatial = self.spatial_gate(m)
        gate = torch.sigmoid(channel + spatial)
        value = self.value_proj(m)

        beta = self.beta_max * torch.sigmoid(self.beta_raw) if self.beta_max > 0 else 0.0
        residual = self.drop_path(gate * value).reshape(g, k, c, h, w)
        out = x + residual * beta
        debug = {
            "tdgx_enabled": True,
            "tdgx_beta": float(beta.detach().cpu()) if torch.is_tensor(beta) else float(beta),
            "tdgx_gate_mean": float(gate.detach().float().mean().cpu()),
            "tdgx_value_norm": float(value.detach().float().norm().cpu()),
        }
        return out, debug
