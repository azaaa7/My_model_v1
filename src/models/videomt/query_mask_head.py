from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(max(1, int(num_layers))):
        src = in_dim if idx == 0 else hidden_dim
        dst = out_dim if idx == max(1, int(num_layers)) - 1 else hidden_dim
        layers.append(nn.Linear(src, dst))
        if idx != max(1, int(num_layers)) - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class QueryMaskHead(nn.Module):
    """Predict query-level masks and aggregate them into final binary logits."""

    def __init__(self, query_dim: int, in_mask_channels: int, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.mask_dim = int(cfg.get("mask_dim", 256))
        self.mask_resolution = int(cfg.get("mask_resolution", 128))
        self.upsample_to_input = bool(cfg.get("upsample_to_input", True))

        self.mask_embed = build_mlp(
            query_dim,
            query_dim,
            self.mask_dim,
            int(cfg.get("mask_embed_mlp_layers", 3)),
        )
        self.mask_feature_proj = nn.Conv2d(in_mask_channels, self.mask_dim, kernel_size=1)

        score_cfg = cfg.get("score_head", {}) or {}
        self.use_score = bool(score_cfg.get("enabled", True))
        score_hidden = int(score_cfg.get("hidden_dim", 256))
        self.score_head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, 1),
        ) if self.use_score else None

        agg_cfg = cfg.get("aggregation", {}) or {}
        self.aggregation = str(agg_cfg.get("type", "logsumexp"))
        self.temperature = float(agg_cfg.get("temperature", 1.0))
        self.use_query_score = bool(agg_cfg.get("use_query_score", True))
        self.normalize_logsumexp = bool(agg_cfg.get("normalize_logsumexp", False))

    def forward(
        self,
        query_states: torch.Tensor,
        mask_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        squeeze_window = False
        if query_states.ndim == 4:
            query_states = query_states[:, None]
            mask_features = mask_features[:, None]
            squeeze_window = True
        if query_states.ndim != 5 or mask_features.ndim != 6:
            raise ValueError(
                f"Expected query_states [B,W,T,Q,C] and mask_features [B,W,T,C,H,W], "
                f"got {tuple(query_states.shape)} / {tuple(mask_features.shape)}"
            )

        b, w, t, q, c = query_states.shape
        _, _, _, cm, hm, wm = mask_features.shape
        q_flat = query_states.reshape(b * w * t, q, c)
        f_flat = mask_features.reshape(b * w * t, cm, hm, wm)

        mask_embed = self.mask_embed(q_flat)
        mask_feat = self.mask_feature_proj(f_flat)
        query_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_feat)
        query_scores = self.score_head(q_flat) if self.score_head is not None else torch.zeros(
            q_flat.shape[0],
            q,
            1,
            device=q_flat.device,
            dtype=q_flat.dtype,
        )

        query_logits_for_agg = query_logits
        if self.use_query_score:
            query_logits_for_agg = query_logits_for_agg + query_scores[..., None]

        if self.aggregation == "logsumexp":
            temp = max(self.temperature, 1.0e-6)
            logits = torch.logsumexp(query_logits_for_agg / temp, dim=1, keepdim=True) * temp
            if self.normalize_logsumexp:
                logits = logits - temp * math.log(max(1, q))
        elif self.aggregation == "max":
            logits = query_logits_for_agg.max(dim=1, keepdim=True).values
        elif self.aggregation == "mean":
            logits = query_logits_for_agg.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown query mask aggregation: {self.aggregation}")

        if self.upsample_to_input and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        query_logits = query_logits.reshape(b, w, t, q, hm, wm)
        query_scores = query_scores.reshape(b, w, t, q, 1)
        logits = logits.reshape(b, w, t, 1, logits.shape[-2], logits.shape[-1])
        if squeeze_window:
            logits = logits[:, 0]
            query_logits = query_logits[:, 0]
            query_scores = query_scores[:, 0]
        return {
            "logits": logits,
            "query_logits": query_logits,
            "query_scores": query_scores,
        }
