from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross attention.

        Args:
            q_tokens: [B, Nq, D]
            kv_tokens: [B, Nk, D]
            attn_mask: bool tensor broadcastable to [B, heads, Nq, Nk], True means valid.
        """
        b, nq, _ = q_tokens.shape
        nk = kv_tokens.shape[1]
        q = self.q_proj(q_tokens).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tokens).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).view(b, nk, self.heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            while attn_mask.ndim < logits.ndim:
                attn_mask = attn_mask.unsqueeze(1)
            logits = logits.masked_fill(~attn_mask, torch.finfo(logits.dtype).min)

        if topk is not None and 0 < int(topk) < nk:
            values, indices = torch.topk(logits, k=int(topk), dim=-1)
            sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits = sparse_logits.scatter(-1, indices, values)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, nq, self.dim)
        out = self.out_proj(out)
        debug = {
            "attn_mean": float(attn.detach().mean().cpu()),
            "attn_max": float(attn.detach().amax().cpu()),
        }
        return out, debug


def frame_lower_triangular_mask(
    batch: int,
    q_frames: int,
    q_tokens_per_frame: int,
    kv_tokens_per_frame: int,
    device,
    mode: str = "lower_triangular",
) -> torch.Tensor:
    q_frame = torch.arange(q_frames, device=device).repeat_interleave(q_tokens_per_frame)
    kv_frame = torch.arange(q_frames, device=device).repeat_interleave(kv_tokens_per_frame)
    if str(mode).lower() in {"none", "bidirectional", "all"}:
        valid = torch.ones(q_frame.numel(), kv_frame.numel(), device=device, dtype=torch.bool)
    elif str(mode).lower() in {"lower", "lower_triangular", "causal"}:
        valid = kv_frame.unsqueeze(0) <= q_frame.unsqueeze(1)
    else:
        raise ValueError(f"Unknown frame mask mode: {mode}")
    return valid.unsqueeze(0).expand(batch, -1, -1).contiguous()


def apply_random_keep_mask(valid_mask: torch.Tensor, keep_prob: float) -> tuple[torch.Tensor, float]:
    if keep_prob >= 1.0:
        return valid_mask, 1.0
    random_keep = torch.rand_like(valid_mask.float()) < float(keep_prob)
    masked = valid_mask & random_keep
    any_valid = masked.any(dim=-1, keepdim=True)
    first_valid = valid_mask.float().argmax(dim=-1, keepdim=True)
    repaired = torch.zeros_like(masked).scatter(-1, first_valid, True)
    masked = torch.where(any_valid, masked, repaired & valid_mask)
    ratio = masked.float().sum() / valid_mask.float().sum().clamp_min(1.0)
    return masked, float(ratio.detach().cpu())

