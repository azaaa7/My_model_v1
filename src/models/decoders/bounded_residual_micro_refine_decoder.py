from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_iml_head import DINOv3IMLHead


def _make_norm2d(norm: str, channels: int) -> nn.Module:
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
    raise ValueError(f"Unsupported 2D norm: {norm}")


def _make_norm3d(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D norm: {norm}")


def _zero_init_conv(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _bounded_delta(raw_delta: torch.Tensor, clip_value: float) -> torch.Tensor:
    if clip_value <= 0:
        return raw_delta
    return float(clip_value) * torch.tanh(raw_delta / float(clip_value))


class BoundedResidualMicroRefineDecoder(nn.Module):
    """Paper-style IML head with bounded low/high-res logit refinement."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}

        self.in_channels = int(cfg.get("in_channels", 1024))
        self.image_size = int(cfg.get("image_size", cfg.get("input_size", 512)))
        self.norm = str(cfg.get("norm", "gn"))

        base_cfg = dict(cfg)
        base_cfg.setdefault("type", "dinov3_iml_head")
        base_cfg.setdefault("in_channels", self.in_channels)
        base_cfg.setdefault("hidden1", int(cfg.get("hidden1", self.in_channels // 2)))
        base_cfg.setdefault("hidden2", int(cfg.get("hidden2", self.in_channels // 4)))
        base_cfg.setdefault("image_size", self.image_size)
        base_cfg.setdefault("norm", self.norm)
        self.base_head = DINOv3IMLHead(base_cfg)

        mr_cfg = cfg.get("micro_refine", {}) or {}
        self.enabled = bool(mr_cfg.get("enabled", True))
        self.refine_channels = int(mr_cfg.get("channels", 64))
        self.high_res = int(mr_cfg.get("high_res", 128))
        self.delta32_clip = float(mr_cfg.get("delta32_clip", 1.0))
        self.delta128_clip = float(mr_cfg.get("delta128_clip", 0.75))
        self.use_high128 = bool(mr_cfg.get("use_high128", True))
        self.use_prob = bool(mr_cfg.get("use_prob", True))
        self.use_uncertainty = bool(mr_cfg.get("use_uncertainty", True))
        self.detach_coarse_for_refine = bool(mr_cfg.get("detach_coarse_for_refine", False))

        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.refine_channels, kernel_size=1, bias=False),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
        )

        low_in = self.refine_channels + 1
        if self.use_prob:
            low_in += 1
        if self.use_uncertainty:
            low_in += 1

        self.low3d_pre = nn.Sequential(
            nn.Conv3d(low_in, self.refine_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, self.refine_channels),
            nn.GELU(),
        )
        self.low3d_dw = nn.Sequential(
            nn.Conv3d(
                self.refine_channels,
                self.refine_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                groups=self.refine_channels,
                bias=False,
            ),
            _make_norm3d(self.norm, self.refine_channels),
            nn.GELU(),
        )
        self.low3d_out = nn.Conv3d(self.refine_channels, 1, kernel_size=1, bias=True)
        _zero_init_conv(self.low3d_out)

        high_in = self.refine_channels + 1
        if self.use_prob:
            high_in += 1
        if self.use_uncertainty:
            high_in += 1

        self.high2d = nn.Sequential(
            nn.Conv2d(high_in, self.refine_channels, kernel_size=3, padding=1, bias=False),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
            nn.Conv2d(
                self.refine_channels,
                self.refine_channels,
                kernel_size=3,
                padding=1,
                groups=self.refine_channels,
                bias=False,
            ),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
            nn.Conv2d(self.refine_channels, 1, kernel_size=1, bias=True),
        )
        _zero_init_conv(self.high2d[-1])

    @staticmethod
    def _logit_cues(logits: torch.Tensor, use_prob: bool, use_uncertainty: bool) -> list[torch.Tensor]:
        cues = [logits]
        prob = torch.sigmoid(logits)
        if use_prob:
            cues.append(prob)
        if use_uncertainty:
            cues.append(4.0 * prob * (1.0 - prob))
        return cues

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if features.ndim != 4:
            raise ValueError(f"features must be [N,C,H,W], got {tuple(features.shape)}")
        n, c, h, w = features.shape
        out = self.forward_video(features.reshape(n, 1, c, h, w), output_size=output_size)
        return {
            "logits": out["logits"].reshape(n, 1, *out["logits"].shape[-2:]),
            "logits32": out["logits32"].reshape(n, 1, *out["logits32"].shape[-2:]),
            "logits32_coarse": out["logits32_coarse"].reshape(n, 1, *out["logits32_coarse"].shape[-2:]),
            "logits128": out["logits128"].reshape(n, 1, *out["logits128"].shape[-2:]),
            "delta32": out["delta32"].reshape(n, 1, *out["delta32"].shape[-2:]),
            "delta128": out["delta128"].reshape(n, 1, *out["delta128"].shape[-2:]),
            "debug": out["debug"],
        }

    def forward_video(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,K,C,H,W], got {tuple(features.shape)}")
        if output_size is None:
            output_size = (self.image_size, self.image_size)

        b, k, c, fh, fw = features.shape
        flat_features = features.reshape(b * k, c, fh, fw)
        base = self.base_head(flat_features, output_size=(fh, fw))
        logits32_coarse = base["logits32"].reshape(b, k, 1, fh, fw)

        if not self.enabled:
            logits32 = logits32_coarse
            logits128 = F.interpolate(
                logits32.reshape(b * k, 1, fh, fw),
                size=(self.high_res, self.high_res),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, k, 1, self.high_res, self.high_res)
            logits = F.interpolate(
                logits128.reshape(b * k, 1, self.high_res, self.high_res),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(b, k, 1, *output_size)
            zero32 = torch.zeros_like(logits32)
            zero128 = torch.zeros_like(logits128)
            return {
                "logits": logits,
                "logits32": logits32,
                "logits32_coarse": logits32_coarse,
                "logits128": logits128,
                "delta32": zero32,
                "delta128": zero128,
                "debug": {
                    "decoder_type": "BoundedResidualMicroRefineDecoder",
                    "micro_refine_enabled": False,
                    "feature_shape": tuple(features.shape),
                },
            }

        feat32 = self.feat_proj(flat_features).reshape(b, k, self.refine_channels, fh, fw)
        coarse_for_refine = logits32_coarse.detach() if self.detach_coarse_for_refine else logits32_coarse
        low_cues = self._logit_cues(coarse_for_refine, self.use_prob, self.use_uncertainty)
        low_in = torch.cat([feat32, *low_cues], dim=2).permute(0, 2, 1, 3, 4).contiguous()
        raw_delta32 = self.low3d_out(self.low3d_dw(self.low3d_pre(low_in)))
        delta32 = _bounded_delta(raw_delta32, self.delta32_clip).permute(0, 2, 1, 3, 4).contiguous()
        logits32 = logits32_coarse + delta32

        logits128_base = F.interpolate(
            logits32.reshape(b * k, 1, fh, fw),
            size=(self.high_res, self.high_res),
            mode="bilinear",
            align_corners=False,
        )

        if self.use_high128:
            feat128 = F.interpolate(
                feat32.reshape(b * k, self.refine_channels, fh, fw),
                size=(self.high_res, self.high_res),
                mode="bilinear",
                align_corners=False,
            )
            high_cues = self._logit_cues(logits128_base, self.use_prob, self.use_uncertainty)
            raw_delta128 = self.high2d(torch.cat([feat128, *high_cues], dim=1))
            delta128_flat = _bounded_delta(raw_delta128, self.delta128_clip)
            logits128_flat = logits128_base + delta128_flat
        else:
            delta128_flat = torch.zeros_like(logits128_base)
            logits128_flat = logits128_base

        logits = F.interpolate(
            logits128_flat,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, k, 1, *output_size)
        logits128 = logits128_flat.reshape(b, k, 1, self.high_res, self.high_res)
        delta128 = delta128_flat.reshape(b, k, 1, self.high_res, self.high_res)

        return {
            "logits": logits,
            "logits32": logits32,
            "logits32_coarse": logits32_coarse,
            "logits128": logits128,
            "delta32": delta32,
            "delta128": delta128,
            "debug": {
                "decoder_type": "BoundedResidualMicroRefineDecoder",
                "micro_refine_enabled": True,
                "feature_shape": tuple(features.shape),
                "logits32_coarse_shape": tuple(logits32_coarse.shape),
                "logits32_shape": tuple(logits32.shape),
                "logits128_shape": tuple(logits128.shape),
                "logits_shape": tuple(logits.shape),
                "delta32_clip": self.delta32_clip,
                "delta128_clip": self.delta128_clip,
                "delta32_abs_mean": float(delta32.detach().abs().mean().cpu()),
                "delta128_abs_mean": float(delta128.detach().abs().mean().cpu()),
                "delta32_abs_max": float(delta32.detach().abs().max().cpu()),
                "delta128_abs_max": float(delta128.detach().abs().max().cpu()),
            },
        }
