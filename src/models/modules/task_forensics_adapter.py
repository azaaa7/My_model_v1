from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableClampedScalar(nn.Module):
    def __init__(self, init: float = 0.001, max_value: float = 0.04):
        super().__init__()
        max_value = max(float(max_value), 1.0e-8)
        ratio = min(max(float(init) / max_value, 1.0e-4), 1.0 - 1.0e-4)
        self.raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.max_value = float(max_value)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) * self.max_value


class FixedResidualStem(nn.Module):
    def __init__(self, target_resolution: int = 32, detach_input: bool = True):
        super().__init__()
        self.target_resolution = int(target_resolution)
        self.detach_input = bool(detach_input)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = sobel_x.t()
        self.register_buffer("lap", lap.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_x", sobel_x.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.reshape(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _gray(frames: torch.Tensor) -> torch.Tensor:
        return 0.299 * frames[:, :, 0:1] + 0.587 * frames[:, :, 1:2] + 0.114 * frames[:, :, 2:3]

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5:
            raise ValueError(f"rgb must be [B,T,3,H,W], got {tuple(rgb.shape)}")
        x = rgb.detach() if self.detach_input else rgb
        b, t, _c, h, w = x.shape
        gray = self._gray(x).reshape(b * t, 1, h, w).float()
        lap = F.conv2d(gray, self.lap.to(dtype=gray.dtype), padding=1)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        local = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        residual = gray - local
        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1.0e-6)
        temporal = torch.zeros_like(gray)
        if t > 1:
            seq = gray.reshape(b, t, 1, h, w)
            temporal = (seq - seq.mean(dim=1, keepdim=True)).reshape(b * t, 1, h, w)
        out = torch.cat([gray, lap, gx, gy, residual, grad_mag, temporal], dim=1)
        return F.adaptive_avg_pool2d(out, (self.target_resolution, self.target_resolution))


class TaskSpecificForensicsAdapter(nn.Module):
    """Low-alpha post-backbone adapter for task-specific inpainting traces."""

    def __init__(self, in_dim: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.in_dim = int(in_dim)
        self.adapter_dim = int(cfg.get("adapter_dim", 64))
        self.num_prompt_tokens = int(cfg.get("num_prompt_tokens", 4))
        self.target_resolution = int(cfg.get("target_resolution", 32))
        self.drop_path = float(cfg.get("drop_path", 0.15))
        self.warmup_epochs = int(cfg.get("warmup_epochs", 10))
        self.alpha = LearnableClampedScalar(float(cfg.get("alpha_init", 0.001)), float(cfg.get("alpha_max", 0.04)))

        self.residual_stem = FixedResidualStem(self.target_resolution, bool(cfg.get("detach_input", True)))
        self.res_proj = nn.Sequential(
            nn.Conv2d(7, self.adapter_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.adapter_dim, self.adapter_dim, 3, padding=1, groups=max(1, self.adapter_dim), bias=False),
            nn.GroupNorm(max(1, min(8, self.adapter_dim)), self.adapter_dim),
            nn.GELU(),
        )
        self.prompt_tokens = nn.Parameter(torch.randn(self.num_prompt_tokens, self.adapter_dim) * 0.02)
        self.q_proj = nn.Sequential(nn.GroupNorm(32, self.in_dim), nn.Conv2d(self.in_dim, self.adapter_dim, 1))
        self.k_proj = nn.Conv2d(self.adapter_dim, self.adapter_dim, 1)
        self.v_proj = nn.Conv2d(self.adapter_dim, self.adapter_dim, 1)
        self.out_proj = nn.Conv2d(self.adapter_dim, self.in_dim, 1)
        self.gate = nn.Conv2d(self.in_dim + self.adapter_dim, 1, 1)
        self.mask_head = nn.Conv2d(self.adapter_dim, 1, 1)
        self.boundary_head = nn.Conv2d(self.adapter_dim, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate.bias, float(cfg.get("gate_bias_init", -2.0)))

    def _attention(self, feature: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature.shape
        q = self.q_proj(feature).flatten(2).transpose(1, 2)
        k_map = self.k_proj(residual)
        v_map = self.v_proj(residual)
        k = k_map.flatten(2).transpose(1, 2)
        v = v_map.flatten(2).transpose(1, 2)
        prompt = self.prompt_tokens.to(device=feature.device, dtype=feature.dtype)[None].expand(b, -1, -1)
        k = torch.cat([k, prompt], dim=1)
        v = torch.cat([v, prompt], dim=1)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.adapter_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, self.adapter_dim, h, w)
        return out

    def forward(self, features: torch.Tensor, rgb: torch.Tensor, epoch: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor | dict[str, float]]]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,T,C,H,W], got {tuple(features.shape)}")
        b, t, c, h, w = features.shape
        flat_feat = features.reshape(b * t, c, h, w)
        residual = self.residual_stem(rgb).to(dtype=flat_feat.dtype)
        res_feat = self.res_proj(residual)
        ctx = self._attention(flat_feat, res_feat)
        gate = torch.sigmoid(self.gate(torch.cat([flat_feat, ctx], dim=1)))
        if self.training and self.drop_path > 0.0:
            keep = torch.rand(b * t, 1, 1, 1, device=flat_feat.device, dtype=flat_feat.dtype) >= self.drop_path
            gate = gate * keep
        alpha = self.alpha()
        if self.training and epoch is not None and self.warmup_epochs > 0:
            alpha = alpha * min(1.0, max(0.0, float(epoch + 1) / float(self.warmup_epochs)))
        delta = alpha.to(dtype=flat_feat.dtype) * gate * self.out_proj(ctx)
        mask32 = self.mask_head(ctx).reshape(b, t, 1, h, w)
        boundary32 = self.boundary_head(ctx).reshape(b, t, 1, h, w)
        zero_aux_dep = (mask32.sum() + boundary32.sum()) * 0.0
        out = (flat_feat + delta + zero_aux_dep).reshape(b, t, c, h, w)
        aux = {
            "adapter_mask32": mask32,
            "adapter_boundary32": boundary32,
            "adapter_gate": gate.reshape(b, t, 1, h, w),
            "adapter_alpha_tensor": alpha.reshape(()),
            "debug": {
                "adapter_alpha": float(alpha.detach().cpu()),
                "adapter_gate_mean": float(gate.detach().float().mean().cpu()),
                "adapter_delta_norm": float(delta.detach().float().norm().cpu()),
            },
        }
        return out, aux
