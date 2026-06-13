from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class QueryPatchBlock(nn.Module):
    """Forgery query and patch-token interaction block."""

    def __init__(self, dim: int = 1024, num_heads: int = 8, ffn_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        hidden = int(dim * ffn_ratio)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        q_norm = self.q_norm(queries)
        kv_norm = self.kv_norm(patch_tokens)
        cross, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)
        queries = queries + cross

        q_norm = self.self_norm(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        queries = queries + self_out

        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class WindowQueryFusion(nn.Module):
    """VidEoMT-style forward/backward query propagation over one window."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.dim = int(cfg.get("dim", 1024))
        self.num_queries = int(cfg.get("num_queries", 16))
        self.bidirectional = bool(cfg.get("bidirectional", True))

        heads = int(cfg.get("heads", cfg.get("num_heads", 8)))
        ffn_ratio = float(cfg.get("ffn_ratio", 4.0))
        dropout = float(cfg.get("dropout", 0.0))

        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.dim) * 0.02)
        self.forward_linear = nn.Linear(self.dim, self.dim)
        self.backward_linear = nn.Linear(self.dim, self.dim)
        self.forward_block = QueryPatchBlock(self.dim, heads, ffn_ratio, dropout)
        self.backward_block = QueryPatchBlock(self.dim, heads, ffn_ratio, dropout)
        self.query_to_feature = nn.Linear(self.dim, self.dim)
        self.residual_alpha = nn.Parameter(torch.tensor(float(cfg.get("residual_alpha_init", 0.0))))

    def _run_forward(self, patch_tokens: torch.Tensor, q_lrn: torch.Tensor) -> torch.Tensor:
        _b, num_frames, _n, _c = patch_tokens.shape
        outputs = []
        prev = None
        for frame_idx in range(num_frames):
            q_in = q_lrn if prev is None else self.forward_linear(prev) + q_lrn
            q_out = self.forward_block(q_in, patch_tokens[:, frame_idx])
            outputs.append(q_out)
            prev = q_out
        return torch.stack(outputs, dim=1)

    def _run_backward(self, patch_tokens: torch.Tensor, q_lrn: torch.Tensor) -> torch.Tensor:
        _b, num_frames, _n, _c = patch_tokens.shape
        outputs: list[torch.Tensor | None] = [None] * num_frames
        next_q = None
        for frame_idx in reversed(range(num_frames)):
            q_in = q_lrn if next_q is None else self.backward_linear(next_q) + q_lrn
            q_out = self.backward_block(q_in, patch_tokens[:, frame_idx])
            outputs[frame_idx] = q_out
            next_q = q_out
        return torch.stack([x for x in outputs if x is not None], dim=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | tuple[int, ...]]]:
        if features.ndim != 5:
            raise ValueError(f"WindowQueryFusion expects [B,T,C,H,W], got {tuple(features.shape)}")
        batch, num_frames, channels, feat_h, feat_w = features.shape
        if channels != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {channels}")

        patch_tokens = features.flatten(3).transpose(2, 3).contiguous()
        q_lrn = self.learned_queries.unsqueeze(0).expand(batch, -1, -1)
        q_forward = self._run_forward(patch_tokens, q_lrn)
        if self.bidirectional:
            q_backward = self._run_backward(patch_tokens, q_lrn)
            query_states = 0.5 * (q_forward + q_backward)
        else:
            query_states = q_forward

        query_context = query_states.mean(dim=2)
        query_context = self.query_to_feature(query_context).reshape(batch, num_frames, channels, 1, 1)
        enhanced = features + self.residual_alpha.to(dtype=features.dtype) * query_context.to(dtype=features.dtype)

        aux = {
            "query_states": query_states,
            "videomt_query_alpha": float(self.residual_alpha.detach().cpu()),
            "videomt_query_shape": tuple(query_states.shape),
        }
        return enhanced, aux
