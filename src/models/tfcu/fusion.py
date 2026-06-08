from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LowResTFCUFusion(nn.Module):
    def __init__(self, in_channels: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.in_channels = int(in_channels)
        self.static_channels = int(cfg.get("static_channels", 128))
        self.ccm_channels = int(cfg.get("ccm_channels", 64))
        self.fgm_channels = int(cfg.get("fgm_channels", 64))
        self.out_channels = int(cfg.get("out_channels", 128))

        self.x_proj = nn.Sequential(nn.GroupNorm(32, self.in_channels), nn.Conv2d(self.in_channels, self.static_channels, 1))
        self.cc_proj = nn.Sequential(nn.GroupNorm(32, self.in_channels), nn.Conv2d(self.in_channels, self.ccm_channels, 1))
        self.ip_proj = nn.Sequential(nn.GroupNorm(32, self.in_channels), nn.Conv2d(self.in_channels, self.fgm_channels, 1))
        self.fuse = nn.Sequential(
            nn.Conv2d(self.static_channels + self.ccm_channels + self.fgm_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )
        self.use_depthwise_refine = bool(cfg.get("use_depthwise_refine", True))
        self.refine = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=self.out_channels, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, 1),
        ) if self.use_depthwise_refine else nn.Identity()

    def forward(self, x: torch.Tensor, f_cc: torch.Tensor, f_ip: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        b, k, c, h, w = x.shape
        xf = x.reshape(b * k, c, h, w)
        cf = f_cc.reshape(b * k, c, h, w)
        ipf = f_ip.reshape(b * k, c, h, w)
        x_p = self.x_proj(xf)
        cc_p = self.cc_proj(cf)
        ip_p = self.ip_proj(ipf)
        fused = self.fuse(torch.cat([x_p, cc_p, ip_p], dim=1))
        fused = fused + self.refine(fused)
        debug = {
            "fusion_static_norm": float(x_p.detach().float().norm().cpu()),
            "fusion_ccm_norm": float(cc_p.detach().float().norm().cpu()),
            "fusion_fgm_norm": float(ip_p.detach().float().norm().cpu()),
            "fusion_out_shape": tuple(fused.shape),
        }
        return fused, debug


class StaticLowResFusion(nn.Module):
    """Parameter-matched-ish static adapter used for no-temporal ablations."""

    def __init__(self, in_channels: int = 1024, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        b, k, c, h, w = x.shape
        out = self.net(x.reshape(b * k, c, h, w))
        return out, {"fusion_static_only_shape": tuple(out.shape), "fusion_static_only_norm": float(out.detach().float().norm().cpu())}

