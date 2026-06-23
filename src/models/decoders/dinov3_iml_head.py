from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


class DINOv3IMLHead(nn.Module):
    """DINOv3-IML paper-style 3-conv segmentation head."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        in_channels = int(cfg.get("in_channels", 1024))
        hidden1 = int(cfg.get("hidden1", in_channels // 2))
        hidden2 = int(cfg.get("hidden2", in_channels // 4))
        norm = str(cfg.get("norm", "bn"))
        self.image_size = int(cfg.get("image_size", cfg.get("input_size", 512)))

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=True),
            _make_norm(norm, hidden1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=True),
            _make_norm(norm, hidden2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1, bias=True),
        )
        self._init_seg_head()

    def _init_seg_head(self) -> None:
        for module in self.seg_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if features.ndim != 4:
            raise ValueError(f"features must be [N,C,H,W], got {tuple(features.shape)}")
        if output_size is None:
            output_size = (self.image_size, self.image_size)

        logits32 = self.seg_head(features)
        logits = F.interpolate(logits32, size=output_size, mode="bilinear", align_corners=False)
        return {
            "logits": logits,
            "logits32": logits32,
            "debug": {
                "decoder_type": "DINOv3IMLHead",
                "decoder_input_shape": tuple(features.shape),
                "decoder_logits32_shape": tuple(logits32.shape),
                "decoder_logits_shape": tuple(logits.shape),
            },
        }
