from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatchTokenFusion(nn.Module):
    """Same-patch cross-frame attention for DINOv3 patch tokens.

    Input/output shape is [B, M, K, N, C]. Attention is only performed along K
    for each fixed (batch, clip, patch-position), so it does not mix spatial
    positions and does not introduce any full-video memory state.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, dim: int | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.mode = str(cfg.get("mode", "same_patch_temporal_attention"))
        self.dim = int(dim or cfg.get("dim", 1024))
        self.bottleneck_dim = int(cfg.get("bottleneck_dim", 128))
        self.heads = int(cfg.get("heads", 4))
        if self.bottleneck_dim % self.heads != 0:
            raise ValueError(f"bottleneck_dim={self.bottleneck_dim} must be divisible by heads={self.heads}")
        self.head_dim = self.bottleneck_dim // self.heads
        self.scale = self.head_dim**-0.5
        self.frame_mask = str(cfg.get("frame_mask", "bidirectional")).lower()
        self.return_residual_energy = bool(cfg.get("return_residual_energy", True))
        self.alpha_max = float(cfg.get("alpha_max", 1.0))

        self.norm = nn.LayerNorm(self.dim)
        self.reduce = nn.Linear(self.dim, self.bottleneck_dim)
        self.qkv = nn.Linear(self.bottleneck_dim, self.bottleneck_dim * 3)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.proj = nn.Linear(self.bottleneck_dim, self.dim)
        if bool(cfg.get("proj_zero_init", True)):
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        self.alpha = nn.Parameter(torch.tensor(float(cfg.get("alpha_init", 0.0))))

    def alpha_value(self) -> torch.Tensor:
        if self.alpha_max > 0:
            return self.alpha.clamp(-self.alpha_max, self.alpha_max)
        return self.alpha

    def _attention_mask(self, k: int, device: torch.device) -> torch.Tensor | None:
        if self.frame_mask in {"causal", "lower_triangular", "lower-triangular"}:
            mask = torch.ones(k, k, device=device, dtype=torch.bool).triu(1)
            return mask
        return None

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ):
        if x.ndim != 5:
            raise ValueError(f"TemporalPatchTokenFusion expects [B,M,K,N,C], got {tuple(x.shape)}")
        b, m, k, n, c = x.shape
        if c != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {c}")

        if not self.enabled or k <= 1:
            energy = x.new_zeros(b, m, k, n) if self.return_residual_energy else None
            debug = {
                "ttf_enabled": bool(self.enabled),
                "ttf_alpha": float(self.alpha_value().detach().cpu()),
                "ttf_residual_energy": 0.0,
                "ttf_temporal_len": int(k),
                "ttf_patch_count": int(n),
            }
            aux = {"ttf_residual_energy": energy} if energy is not None else {}
            return (x, debug, aux) if return_aux else (x, debug)

        seq = x.permute(0, 1, 3, 2, 4).reshape(b * m * n, k, c)
        z = self.reduce(self.norm(seq))
        qkv = self.qkv(z).reshape(seq.shape[0], k, 3, self.heads, self.head_dim)
        q, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn = (q @ key.transpose(-2, -1)) * self.scale
        mask = self._attention_mask(k, x.device)
        if mask is not None:
            attn = attn.masked_fill(mask.view(1, 1, k, k), torch.finfo(attn.dtype).min)
        attn = F.softmax(attn.float(), dim=-1).to(dtype=value.dtype)
        attn = self.dropout(attn)
        residual = (attn @ value).transpose(1, 2).reshape(seq.shape[0], k, self.bottleneck_dim)
        residual = self.dropout(self.proj(residual))
        alpha = self.alpha_value().to(dtype=residual.dtype)
        fused = seq + alpha * residual
        out = fused.reshape(b, m, n, k, c).permute(0, 1, 3, 2, 4).contiguous()

        injected = alpha * residual
        energy_seq = injected.float().pow(2).mean(dim=-1)
        energy = energy_seq.reshape(b, m, n, k).permute(0, 1, 3, 2).contiguous()
        energy_value = float(energy.detach().mean().cpu())
        debug = {
            "ttf_enabled": True,
            "ttf_alpha": float(alpha.detach().cpu()),
            "ttf_residual_energy": energy_value,
            "ttf_temporal_len": int(k),
            "ttf_patch_count": int(n),
        }
        aux = {"ttf_residual_energy": energy} if self.return_residual_energy else {}
        return (out, debug, aux) if return_aux else (out, debug)
