from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNoiseForensicBranch(nn.Module):
    """Low-level residual/noise branch for domain-robust forensic traces.

    The branch intentionally stays small: fixed high-pass filters suppress most
    scene semantics, then a shallow trainable projection maps residual evidence
    to the 32x32 decoder/fusion resolution.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.out_channels = int(cfg.get("out_channels", 32))
        self.target_resolution = int(cfg.get("target_resolution", 32))
        self.fusion_channels = int(cfg.get("fusion_channels", 128))
        self.alpha = nn.Parameter(torch.tensor(float(cfg.get("alpha_init", 0.01))))
        self.detach_input = bool(cfg.get("detach_input", False))

        in_channels = 7
        hidden = int(cfg.get("hidden_channels", 32))
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )
        self.to_fusion = nn.Conv2d(self.out_channels, self.fusion_channels, kernel_size=1)
        if bool(cfg.get("zero_init_fusion", True)):
            nn.init.zeros_(self.to_fusion.weight)
            if self.to_fusion.bias is not None:
                nn.init.zeros_(self.to_fusion.bias)

        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = sobel_x.t()
        self.register_buffer("lap_kernel", lap.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_x_kernel", sobel_x.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y_kernel", sobel_y.reshape(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _gray(frames: torch.Tensor) -> torch.Tensor:
        r, g, b = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _highpass(self, gray: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(gray, self.lap_kernel.to(dtype=gray.dtype), padding=1)
        gx = F.conv2d(gray, self.sobel_x_kernel.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y_kernel.to(dtype=gray.dtype), padding=1)
        local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        residual = gray - local_mean
        grad_mag = torch.sqrt(gx.float().pow(2) + gy.float().pow(2) + 1.0e-6).to(dtype=gray.dtype)
        abs_lap = lap.abs()
        abs_residual = residual.abs()
        return torch.cat([gray, lap, gx, gy, residual, grad_mag, abs_lap + abs_residual], dim=1)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | tuple[int, ...]]]:
        if not self.enabled:
            bmk = frames.shape[0]
            zero = torch.zeros(
                bmk,
                self.fusion_channels,
                self.target_resolution,
                self.target_resolution,
                device=frames.device,
                dtype=frames.dtype,
            )
            return zero, {"forensic_enabled": False}

        x = frames.detach() if self.detach_input else frames
        gray = self._gray(x)
        residual = self._highpass(gray)
        residual = F.adaptive_avg_pool2d(residual, (self.target_resolution, self.target_resolution))
        feat = self.project(residual)
        fusion_feat = self.to_fusion(feat)
        out = self.alpha * fusion_feat
        debug = {
            "forensic_enabled": True,
            "forensic_residual_shape": tuple(residual.shape),
            "forensic_feat_shape": tuple(feat.shape),
            "forensic_alpha": float(self.alpha.detach().cpu()),
            "forensic_norm": float(out.detach().float().norm().cpu()),
        }
        return out, debug
