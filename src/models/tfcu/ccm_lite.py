from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadCrossAttention, apply_random_keep_mask, frame_lower_triangular_mask


class CCMLite(nn.Module):
    """Consecutive-frame masked correlation module."""

    def __init__(self, in_channels: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.in_channels = int(in_channels)
        self.dim = int(cfg.get("dim", 128))
        self.heads = int(cfg.get("heads", 4))
        self.q_resolution = int(cfg.get("q_resolution", 32))
        self.kv_resolution = int(cfg.get("kv_resolution", 16))
        self.frame_mask = str(cfg.get("frame_mask", "lower_triangular"))
        random_cfg = cfg.get("random_mask", {}) or {}
        self.random_mask_enabled = bool(random_cfg.get("enabled", False))
        self.random_keep_prob = float(random_cfg.get("keep_prob", 0.7))
        self.alpha_cc = nn.Parameter(torch.tensor(float(cfg.get("alpha_init", 0.002))))

        self.norm = nn.GroupNorm(32, self.in_channels)
        self.to_dim = nn.Conv2d(self.in_channels, self.dim, kernel_size=1)
        self.attn = MultiHeadCrossAttention(self.dim, heads=self.heads)
        self.fuse = nn.Sequential(
            nn.GroupNorm(32, self.in_channels + self.dim),
            nn.Conv2d(self.in_channels + self.dim, self.in_channels, kernel_size=1),
        )
        self.aux_head = nn.Conv2d(self.dim, 1, kernel_size=1) if bool(cfg.get("aux_head", True)) else None
        if bool(cfg.get("fuse_zero_init", True)):
            conv = self.fuse[-1]
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if not self.enabled:
            z = torch.zeros(x.shape[0], x.shape[1], self.dim, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
            return x, z, None, {"ccm_enabled": False}

        b, k, c, h, w = x.shape
        x_flat = x.reshape(b * k, c, h, w)
        x_norm = self.norm(x_flat)
        x_dim = self.to_dim(x_norm).reshape(b, k, self.dim, h, w)
        kv = F.adaptive_avg_pool2d(x_dim.reshape(b * k, self.dim, h, w), (self.kv_resolution, self.kv_resolution))

        q_tokens = x_dim.permute(0, 1, 3, 4, 2).reshape(b, k * h * w, self.dim)
        kv_tokens = kv.reshape(b, k, self.dim, -1).permute(0, 1, 3, 2).reshape(
            b, k * self.kv_resolution * self.kv_resolution, self.dim
        )

        valid_mask = frame_lower_triangular_mask(
            b,
            q_frames=k,
            q_tokens_per_frame=h * w,
            kv_tokens_per_frame=self.kv_resolution * self.kv_resolution,
            device=x.device,
            mode=self.frame_mask,
        )
        random_keep_ratio = 1.0
        if self.training and self.random_mask_enabled:
            valid_mask, random_keep_ratio = apply_random_keep_mask(valid_mask, self.random_keep_prob)

        attn_out, attn_debug = self.attn(q_tokens, kv_tokens, attn_mask=valid_mask)
        ccm_feat = attn_out.reshape(b, k, h, w, self.dim).permute(0, 1, 4, 2, 3).contiguous()
        fuse_in = torch.cat([x_norm, ccm_feat.reshape(b * k, self.dim, h, w)], dim=1)
        residual = self.fuse(fuse_in).reshape(b, k, c, h, w)
        f_cc = x + self.alpha_cc * residual
        aux_logit = self.aux_head(ccm_feat.reshape(b * k, self.dim, h, w)).reshape(b, k, 1, h, w) if self.aux_head else None

        valid_ratio = valid_mask.float().mean()
        debug = {
            "ccm_enabled": True,
            "ccm_q_shape": tuple(q_tokens.shape),
            "ccm_kv_shape": tuple(kv_tokens.shape),
            "ccm_mask_valid_ratio": float(valid_ratio.detach().cpu()),
            "ccm_random_keep_ratio": random_keep_ratio,
            "ccm_alpha": float(self.alpha_cc.detach().cpu()),
            "ccm_residual_norm": float(residual.detach().float().norm().cpu()),
            **attn_debug,
        }
        return f_cc, ccm_feat, aux_logit, debug

