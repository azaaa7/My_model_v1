from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LiteBoundaryDecoder(nn.Module):
    """32->64->128->256 low-res decoder; 512 is only 1-channel logit upsample."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        in_channels = int(cfg.get("in_channels", 128))
        stages = cfg.get("stages", [
            {"resolution": 64, "channels": 96},
            {"resolution": 128, "channels": 48},
            {"resolution": 256, "channels": 16},
        ])
        ch64 = int(stages[0].get("channels", 96))
        ch128 = int(stages[1].get("channels", 48))
        ch256 = int(stages[2].get("channels", 16))
        self.final_upsample = str(cfg.get("final_upsample", "bilinear"))
        self.boundary_enabled = bool((cfg.get("boundary_head", {}) or {}).get("enabled", True))

        self.up64 = ConvBNAct(in_channels, ch64)
        self.up128 = ConvBNAct(ch64, ch128)
        self.mask_head128 = nn.Conv2d(ch128, 1, 1)
        self.boundary_head128 = nn.Conv2d(ch128, 1, 1) if self.boundary_enabled else None
        self.up256 = ConvBNAct(ch128, ch256)
        self.mask_head256 = nn.Conv2d(ch256, 1, 1)

    def forward(self, f32: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        debug = {"decoder_input_shape": tuple(f32.shape)}
        f64 = F.interpolate(f32, scale_factor=2, mode="bilinear", align_corners=False)
        f64 = self.up64(f64)
        debug["decoder_f64_shape"] = tuple(f64.shape)

        f128 = F.interpolate(f64, scale_factor=2, mode="bilinear", align_corners=False)
        f128 = self.up128(f128)
        debug["decoder_f128_shape"] = tuple(f128.shape)
        mask128 = self.mask_head128(f128)
        boundary128 = self.boundary_head128(f128) if self.boundary_head128 is not None else None

        f256 = F.interpolate(f128, scale_factor=2, mode="bilinear", align_corners=False)
        f256 = self.up256(f256)
        debug["decoder_f256_shape"] = tuple(f256.shape)
        mask256 = self.mask_head256(f256)
        logits = F.interpolate(mask256, scale_factor=2, mode="bilinear", align_corners=False)
        debug["decoder_logit512_shape"] = tuple(logits.shape)
        return {
            "mask128": mask128,
            "boundary128": boundary128,
            "mask256": mask256,
            "logits": logits,
            "debug": debug,
        }

