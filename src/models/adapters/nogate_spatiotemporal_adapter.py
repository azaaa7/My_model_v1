from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _make_norm3d(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def _make_activation(act: str) -> nn.Module:
    act = str(act).lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported act: {act}")


class DepthwiseConv3dBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int, int],
        norm: str = "gn",
        act: str = "gelu",
    ) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            _make_norm3d(norm, channels),
            _make_activation(act),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NoGateSpatiotemporalAdapter(nn.Module):
    """Lightweight no-gate spatiotemporal adapter over clip features."""

    def __init__(self, in_channels: int = 1024, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", True))
        self.in_channels = int(in_channels)
        self.bottleneck = int(cfg.get("bottleneck", 128))
        self.temporal_kernel = int(cfg.get("temporal_kernel", 3))
        self.spatial_kernel = int(cfg.get("spatial_kernel", 3))
        self.multiscale = bool(cfg.get("multiscale", False))
        self.norm = str(cfg.get("norm", "gn"))
        self.act = str(cfg.get("act", "gelu"))
        self.zero_init_up = bool(cfg.get("zero_init_up", True))
        self.drop_path_prob = float(cfg.get("drop_path_prob", 0.0))

        self.down = nn.Sequential(
            nn.Conv3d(self.in_channels, self.bottleneck, kernel_size=1, bias=False),
            _make_norm3d(self.norm, self.bottleneck),
            _make_activation(self.act),
        )

        if self.multiscale:
            spatial_kernels = [3, 5, 7]
            temporal_kernels = [3, 5, 7]
        else:
            spatial_kernels = [self.spatial_kernel]
            temporal_kernels = [self.temporal_kernel]

        self.spatial_blocks = nn.ModuleList(
            [
                DepthwiseConv3dBlock(
                    self.bottleneck,
                    kernel_size=(1, k, k),
                    norm=self.norm,
                    act=self.act,
                )
                for k in spatial_kernels
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                DepthwiseConv3dBlock(
                    self.bottleneck,
                    kernel_size=(k, 1, 1),
                    norm=self.norm,
                    act=self.act,
                )
                for k in temporal_kernels
            ]
        )
        self.up = nn.Conv3d(self.bottleneck, self.in_channels, kernel_size=1, bias=True)

        if self.zero_init_up:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def _drop_path(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob <= 0.0:
            return residual
        keep_prob = 1.0 - self.drop_path_prob
        shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        random_tensor = residual.new_empty(shape).bernoulli_(keep_prob)
        return residual * random_tensor / keep_prob

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return x, {"nogate_sta_enabled": False}
        if x.ndim != 5:
            raise ValueError(f"NoGateSpatiotemporalAdapter expects [B,K,C,H,W], got {tuple(x.shape)}")

        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.down(x_3d)

        spatial = sum(block(z) for block in self.spatial_blocks) / len(self.spatial_blocks)
        temporal = sum(block(z) for block in self.temporal_blocks) / len(self.temporal_blocks)
        fused = 0.5 * (spatial + temporal)

        residual = self._drop_path(self.up(fused))
        out_3d = x_3d + residual
        out = out_3d.permute(0, 2, 1, 3, 4).contiguous()

        debug = {
            "nogate_sta_enabled": True,
            "input_shape": tuple(x.shape),
            "bottleneck": self.bottleneck,
            "multiscale": self.multiscale,
            "zero_init_up": self.zero_init_up,
            "drop_path_prob": self.drop_path_prob,
            "residual_norm": float(residual.detach().float().norm().cpu()),
            "input_norm": float(x_3d.detach().float().norm().cpu()),
        }
        return out, debug
