from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    depth = max(1, int(num_layers))
    for i in range(depth):
        src = in_dim if i == 0 else hidden_dim
        dst = out_dim if i == depth - 1 else hidden_dim
        layers.append(nn.Linear(src, dst))
        if i != depth - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class QueryDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.self_norm(queries + self.self_attn(queries, queries, queries, need_weights=False)[0])
        q = self.cross_norm(q + self.cross_attn(q, tokens, tokens, need_weights=False)[0])
        q = self.ffn_norm(q + self.ffn(q))
        return q


class QueryVolumeDecoder(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.in_channels = int(cfg.get("in_channels", 1024))
        self.query_dim = int(cfg.get("query_dim", 256))
        self.num_queries = int(cfg.get("num_queries", 8))
        self.num_layers = int(cfg.get("num_layers", 3))
        self.num_heads = int(cfg.get("num_heads", 8))
        self.ffn_dim = int(cfg.get("ffn_dim", 1024))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.image_size = int(cfg.get("image_size", 512))
        self.output_resolution = int(cfg.get("output_resolution", 32))
        self.use_aux_outputs = bool(cfg.get("use_aux_outputs", True))
        self.combine_method = str(cfg.get("combine_method", "soft_or")).lower()
        self.max_frames = int(cfg.get("max_frames", 8))
        self.max_hw = int(cfg.get("max_hw", self.output_resolution * self.output_resolution))

        self.input_proj = nn.Conv2d(self.in_channels, self.query_dim, kernel_size=1, bias=False)
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, self.query_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_frames, self.query_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.max_hw, self.query_dim))
        self.layers = nn.ModuleList(
            [QueryDecoderLayer(self.query_dim, self.num_heads, self.ffn_dim, self.dropout) for _ in range(self.num_layers)]
        )
        self.mask_embed = _make_mlp(self.query_dim, self.query_dim, self.query_dim, 3)
        self.pixel_embed = nn.Conv2d(self.query_dim, self.query_dim, kernel_size=1)
        self.score_head = nn.Linear(self.query_dim, 1)
        nn.init.zeros_(self.score_head.bias)

    def _combine_queries(self, mask_logits_q: torch.Tensor, score_logits: torch.Tensor) -> torch.Tensor:
        if self.combine_method != "soft_or":
            raise ValueError(f"Unsupported combine_method: {self.combine_method}")
        score_prob = torch.sigmoid(score_logits)
        mask_prob_q = torch.sigmoid(mask_logits_q)
        weighted = mask_prob_q * score_prob[:, :, None, None, None]
        prob = 1.0 - torch.prod(1.0 - weighted, dim=1)
        prob = prob.clamp(1.0e-4, 1.0 - 1.0e-4)
        return torch.log(prob / (1.0 - prob))

    def _predict_layer(self, queries: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, k, d, h, w = feat.shape
        mask_embed = self.mask_embed(queries)
        pixel_embed = self.pixel_embed(feat.reshape(g * k, d, h, w)).reshape(g, k, d, h, w)
        mask_logits_q = torch.einsum("gqd,gkdhw->gqkhw", mask_embed, pixel_embed)
        score_logits = self.score_head(queries).squeeze(-1)
        logits32 = self._combine_queries(mask_logits_q, score_logits).unsqueeze(2)
        return logits32, mask_logits_q, score_logits

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"QueryVolumeDecoder expects [G,K,C,H,W], got {tuple(x.shape)}")
        g, k, c, h, w = x.shape
        feat = self.input_proj(x.reshape(g * k, c, h, w)).reshape(g, k, self.query_dim, h, w)
        tokens = feat.permute(0, 1, 3, 4, 2).reshape(g, k * h * w, self.query_dim)
        spatial = self.spatial_pos[:, : h * w].repeat(1, k, 1)
        temporal = self.temporal_pos[:, :k].repeat_interleave(h * w, dim=1)
        tokens = tokens + spatial + temporal
        queries = self.query_embed.expand(g, -1, -1)

        aux_logits32: list[torch.Tensor] = []
        aux_logits: list[torch.Tensor] = []
        last_query_logits = None
        last_query_scores = None
        for layer_idx, layer in enumerate(self.layers):
            queries = layer(queries, tokens)
            logits32, query_logits, query_scores = self._predict_layer(queries, feat)
            last_query_logits = query_logits
            last_query_scores = query_scores
            if self.use_aux_outputs and layer_idx < len(self.layers) - 1:
                aux_logits32.append(logits32)
                aux_logits.append(
                    F.interpolate(
                        logits32.reshape(g * k, 1, h, w),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(g, k, 1, self.image_size, self.image_size)
                )

        logits = F.interpolate(
            logits32.reshape(g * k, 1, h, w),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(g, k, 1, self.image_size, self.image_size)
        return {
            "logits": logits,
            "logits32": logits32,
            "aux_logits": aux_logits,
            "aux_logits32": aux_logits32,
            "query_logits": last_query_logits,
            "query_scores": last_query_scores,
            "debug": {
                "decoder_type": "QueryVolumeDecoder",
                "decoder_input_shape": tuple(x.shape),
                "decoder_logits_shape": tuple(logits.shape),
                "decoder_logits32_shape": tuple(logits32.shape),
                "num_queries": self.num_queries,
                "num_layers": self.num_layers,
            },
        }
