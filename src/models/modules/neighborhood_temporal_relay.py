from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 1.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedRelativePositionBias3D(nn.Module):
    def __init__(self, num_heads: int, temporal_radius: int, spatial_radius: int):
        super().__init__()
        self.temporal_radius = int(temporal_radius)
        self.spatial_radius = int(spatial_radius)
        self.relative_bias = nn.Parameter(
            torch.zeros(
                num_heads,
                2 * self.temporal_radius + 1,
                2 * self.spatial_radius + 1,
                2 * self.spatial_radius + 1,
            )
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

        offsets = []
        for dt in range(-self.temporal_radius, self.temporal_radius + 1):
            for dy in range(-self.spatial_radius, self.spatial_radius + 1):
                for dx in range(-self.spatial_radius, self.spatial_radius + 1):
                    offsets.append((dt + self.temporal_radius, dy + self.spatial_radius, dx + self.spatial_radius))
        self.register_buffer("offset_index", torch.tensor(offsets, dtype=torch.long), persistent=False)

    def forward(self) -> torch.Tensor:
        idx = self.offset_index
        return self.relative_bias[:, idx[:, 0], idx[:, 1], idx[:, 2]]


class LocalSpatiotemporalNeighborhoodAttention(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dim = int(cfg.get("dim", 1024))
        self.dim = dim
        self.temporal_radius = int(cfg.get("temporal_radius", 1))
        self.spatial_radius = int(cfg.get("spatial_radius", 1))
        self.spatial_dilation = int(cfg.get("spatial_dilation", 1))
        self.num_heads = int(cfg.get("num_heads", 8))
        self.head_dim = dim // self.num_heads
        if self.head_dim * self.num_heads != dim:
            raise ValueError(f"dim={dim} must be divisible by num_heads={self.num_heads}")
        self.scale = self.head_dim ** -0.5
        self.kernel_size = 2 * self.spatial_radius + 1
        self.candidate_count = (2 * self.temporal_radius + 1) * self.kernel_size * self.kernel_size
        self.token_chunk_size = int(cfg.get("token_chunk_size", 256) or 0)

        qkv_bias = bool(cfg.get("qkv_bias", True))
        self.qkv_norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(float(cfg.get("attn_dropout", 0.0)))
        self.proj_dropout = nn.Dropout(float(cfg.get("proj_dropout", 0.0)))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ratio=float(cfg.get("ffn_ratio", 1.0)), dropout=float(cfg.get("proj_dropout", 0.0)))
        self.relative_bias = (
            LearnedRelativePositionBias3D(self.num_heads, self.temporal_radius, self.spatial_radius)
            if bool(cfg.get("relative_position_bias", True))
            else None
        )

    def _spatial_neighbors(self, x_tokens: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, c = x_tokens.shape
        x_bt = x_tokens.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
        patches = F.unfold(
            x_bt,
            kernel_size=self.kernel_size,
            dilation=self.spatial_dilation,
            padding=self.spatial_radius * self.spatial_dilation,
            stride=1,
        )
        patches = patches.transpose(1, 2).reshape(b, t, n, self.kernel_size * self.kernel_size, c)

        ones = torch.ones(1, 1, h, w, device=x_tokens.device, dtype=x_tokens.dtype)
        valid = F.unfold(
            ones,
            kernel_size=self.kernel_size,
            dilation=self.spatial_dilation,
            padding=self.spatial_radius * self.spatial_dilation,
            stride=1,
        )
        valid = valid.transpose(1, 2).reshape(1, 1, n, self.kernel_size * self.kernel_size) > 0.5
        return patches, valid

    def _neighbors(self, x_tokens: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, c = x_tokens.shape
        spatial, spatial_valid = self._spatial_neighbors(x_tokens, h, w)
        chunks = []
        masks = []
        for dt in range(-self.temporal_radius, self.temporal_radius + 1):
            idx = torch.arange(t, device=x_tokens.device) + dt
            time_valid = (idx >= 0) & (idx < t)
            idx = idx.clamp(0, t - 1)
            chunks.append(spatial[:, idx])
            masks.append(spatial_valid.expand(b, t, n, -1) & time_valid.view(1, t, 1, 1))
        return torch.cat(chunks, dim=3), torch.cat(masks, dim=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        b, t, c, h, w = x.shape
        n = h * w
        center = x.permute(0, 1, 3, 4, 2).reshape(b, t, n, c)
        spatial, spatial_valid = self._spatial_neighbors(center, h, w)

        q_all = self.qkv_proj(self.qkv_norm(center))[..., :c].reshape(b, t, n, self.num_heads, self.head_dim)
        rel_bias = None
        if self.relative_bias is not None:
            rel_bias = self.relative_bias().to(device=x.device, dtype=x.dtype).view(1, 1, 1, self.num_heads, -1)

        chunk_size = self.token_chunk_size if self.token_chunk_size > 0 else n
        local_delta_chunks = []
        entropy_sum = center.new_tensor(0.0, dtype=torch.float32)
        attn_max = center.new_tensor(0.0, dtype=torch.float32)
        stat_count = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            neigh_chunks = []
            mask_chunks = []
            for dt in range(-self.temporal_radius, self.temporal_radius + 1):
                idx = torch.arange(t, device=x.device) + dt
                time_valid = (idx >= 0) & (idx < t)
                idx = idx.clamp(0, t - 1)
                neigh_chunks.append(spatial[:, idx, start:end])
                mask_chunks.append(
                    spatial_valid[:, :, start:end].expand(b, t, end - start, -1)
                    & time_valid.view(1, t, 1, 1)
                )
            neighbors = torch.cat(neigh_chunks, dim=3)
            valid_mask = torch.cat(mask_chunks, dim=3)
            q = q_all[:, :, start:end]
            neighbor_qkv = self.qkv_proj(self.qkv_norm(neighbors))
            k = neighbor_qkv[..., c:2 * c].reshape(b, t, end - start, self.candidate_count, self.num_heads, self.head_dim)
            v = neighbor_qkv[..., 2 * c:].reshape(b, t, end - start, self.candidate_count, self.num_heads, self.head_dim)
            k = k.permute(0, 1, 2, 4, 3, 5)
            v = v.permute(0, 1, 2, 4, 3, 5)
            logits = (q.unsqueeze(-2) * k).sum(dim=-1) * self.scale
            if rel_bias is not None:
                logits = logits + rel_bias
            logits = logits.masked_fill(~valid_mask.unsqueeze(3), -1.0e4)
            attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
            attn = self.attn_dropout(attn)
            out = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(b, t, end - start, c)
            local_delta_chunks.append(self.proj_dropout(self.out_proj(out)))
            attn_f = attn.detach().float()
            entropy_sum = entropy_sum + (-(attn_f * attn_f.clamp_min(1.0e-8).log()).sum(dim=-1).sum())
            attn_max = torch.maximum(attn_max, attn_f.amax())
            stat_count += attn_f[..., 0].numel()

        local_delta = torch.cat(local_delta_chunks, dim=2)
        tokens = center + local_delta
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        out_x = tokens.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3).contiguous()

        entropy = entropy_sum / max(stat_count, 1)
        debug = {
            "local_candidate_count": int(self.candidate_count),
            "local_temporal_radius": int(self.temporal_radius),
            "local_spatial_radius": int(self.spatial_radius),
            "local_delta_ratio": float((local_delta.detach().float().norm() / (center.detach().float().norm() + 1.0e-6)).cpu()),
            "local_attention_entropy": float(entropy.detach().cpu()),
            "local_attention_max": float(attn_max.detach().cpu()),
        }
        return out_x, debug


class TemporalRotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, time_index: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("n,d->nd", time_index.to(dtype=self.inv_freq.dtype), self.inv_freq)
        cos = freqs.cos().to(device=x.device, dtype=x.dtype)
        sin = freqs.sin().to(device=x.device, dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos.view(1, 1, x.shape[-2], -1)
        sin = sin.view(1, 1, x.shape[-2], -1)
        return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class RelayTemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, temporal_rope: bool = True):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.attn = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.rope = TemporalRotaryEmbedding(self.head_dim) if temporal_rope else None

    def forward(self, x: torch.Tensor, time_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is not None:
            b, n, c = x.shape
            q = x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            q = self.rope(q, time_index).transpose(1, 2).reshape(b, n, c)
            out, weights = self.attn(q, q, x, need_weights=True, average_attn_weights=False)
        else:
            out, weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        return out, weights


class GlobalTemporalRelayLayer(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dim = int(cfg.get("dim", 1024))
        heads = int(cfg.get("num_heads", 8))
        dropout = float(cfg.get("dropout", 0.0))
        self.relay_from_patches_norm = nn.LayerNorm(dim)
        self.patch_for_relay_norm = nn.LayerNorm(dim)
        self.relay_from_patches = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.relay_ffn_norm = nn.LayerNorm(dim)
        self.relay_ffn = FeedForward(dim, ratio=float(cfg.get("ffn_ratio", 1.0)), dropout=dropout)
        self.relay_temporal_norm = nn.LayerNorm(dim)
        self.relay_temporal = RelayTemporalSelfAttention(dim, heads, dropout, bool(cfg.get("temporal_rope", True)))
        self.relay_temporal_ffn_norm = nn.LayerNorm(dim)
        self.relay_temporal_ffn = FeedForward(dim, ratio=float(cfg.get("ffn_ratio", 1.0)), dropout=dropout)
        self.patch_from_relay_norm = nn.LayerNorm(dim)
        self.relay_for_patch_norm = nn.LayerNorm(dim)
        self.patch_from_relay = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.patch_ffn_norm = nn.LayerNorm(dim)
        self.patch_ffn = FeedForward(dim, ratio=float(cfg.get("ffn_ratio", 1.0)), dropout=dropout)

    def forward(self, tokens: torch.Tensor, relay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        b, t, n, c = tokens.shape
        r = relay.shape[2]
        relay_bt = relay.reshape(b * t, r, c)
        tokens_bt = tokens.reshape(b * t, n, c)
        relay_delta, absorb_attn = self.relay_from_patches(
            self.relay_from_patches_norm(relay_bt),
            self.patch_for_relay_norm(tokens_bt),
            self.patch_for_relay_norm(tokens_bt),
            need_weights=True,
            average_attn_weights=False,
        )
        relay_bt = relay_bt + relay_delta
        relay_bt = relay_bt + self.relay_ffn(self.relay_ffn_norm(relay_bt))
        relay = relay_bt.reshape(b, t, r, c)

        relay_seq = relay.reshape(b, t * r, c)
        time_index = torch.arange(t, device=tokens.device).repeat_interleave(r)
        temporal_delta, temporal_attn = self.relay_temporal(self.relay_temporal_norm(relay_seq), time_index)
        relay_seq = relay_seq + temporal_delta
        relay_seq = relay_seq + self.relay_temporal_ffn(self.relay_temporal_ffn_norm(relay_seq))
        relay = relay_seq.reshape(b, t, r, c)

        relay_delta_to_patch, inject_attn = self.patch_from_relay(
            self.patch_from_relay_norm(tokens_bt),
            self.relay_for_patch_norm(relay.reshape(b * t, r, c)),
            self.relay_for_patch_norm(relay.reshape(b * t, r, c)),
            need_weights=True,
            average_attn_weights=False,
        )
        out_tokens = tokens_bt + relay_delta_to_patch
        out_tokens = out_tokens + self.patch_ffn(self.patch_ffn_norm(out_tokens))
        out_tokens = out_tokens.reshape(b, t, n, c)

        relay_attn = temporal_attn.detach().float()
        relay_entropy = -(relay_attn * relay_attn.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        debug = {
            "relay_token_norm": float(relay.detach().float().norm().cpu()),
            "relay_delta_ratio": float((relay_delta_to_patch.detach().float().norm() / (tokens_bt.detach().float().norm() + 1.0e-6)).cpu()),
            "relay_attention_entropy": float(relay_entropy.cpu()),
            "relay_absorb_attention_max": float(absorb_attn.detach().float().amax().cpu()),
            "relay_inject_attention_max": float(inject_attn.detach().float().amax().cpu()),
        }
        return out_tokens, relay, debug


class NeighborhoodTemporalRelayFusion(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg or {}
        dim = int(self.cfg.get("dim", 1024))
        self.dim = dim
        local_cfg = dict((self.cfg.get("local_neighborhood", {}) or {}))
        relay_cfg = dict((self.cfg.get("global_relay", {}) or {}))
        local_cfg.setdefault("dim", dim)
        relay_cfg.setdefault("dim", dim)
        self.local_enabled = bool(local_cfg.get("enabled", True))
        self.global_enabled = bool(relay_cfg.get("enabled", True))
        self.num_layers = int(relay_cfg.get("num_layers", 1))
        if self.num_layers < 1 or self.num_layers > 2:
            raise ValueError(f"temporal_relay.global_relay.num_layers must be 1 or 2, got {self.num_layers}")
        self.local_layers = nn.ModuleList([LocalSpatiotemporalNeighborhoodAttention(local_cfg) for _ in range(self.num_layers)])
        self.relay_layers = nn.ModuleList([GlobalTemporalRelayLayer(relay_cfg) for _ in range(self.num_layers)])
        self.num_relay_tokens = int(relay_cfg.get("num_tokens", 2))
        self.relay_tokens = nn.Parameter(torch.empty(1, 1, self.num_relay_tokens, dim))
        nn.init.trunc_normal_(self.relay_tokens, std=0.02)
        self.relay_identity = (
            nn.Parameter(torch.empty(1, 1, self.num_relay_tokens, dim))
            if bool(relay_cfg.get("relay_identity_embedding", True))
            else None
        )
        if self.relay_identity is not None:
            nn.init.trunc_normal_(self.relay_identity, std=0.02)
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.fusion_proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        nn.init.xavier_uniform_(self.fusion_proj[0].weight)
        nn.init.zeros_(self.fusion_proj[0].bias)

    def forward(
        self,
        features: torch.Tensor,
        disable_temporal: bool = False,
        disable_local_neighborhood: bool = False,
        disable_global_relay: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,T,C,H,W], got {tuple(features.shape)}")
        b, t, c, h, w = features.shape
        debug: dict[str, Any] = {
            "temporal_relay_disabled": bool(disable_temporal),
            "local_neighborhood_disabled": bool(disable_local_neighborhood),
            "global_relay_disabled": bool(disable_global_relay),
            "local_per_layer": True,
        }
        if self.local_layers:
            debug.update({
                "local_candidate_count": int(self.local_layers[0].candidate_count),
                "local_temporal_radius": int(self.local_layers[0].temporal_radius),
                "local_spatial_radius": int(self.local_layers[0].spatial_radius),
                "local_delta_ratio": 0.0,
                "local_attention_entropy": 0.0,
                "local_attention_max": 0.0,
            })
        debug.update({
            "relay_token_norm": 0.0,
            "relay_delta_ratio": 0.0,
            "relay_attention_entropy": 0.0,
        })
        if disable_temporal or (not self.local_enabled and not self.global_enabled):
            debug["temporal_output_std"] = float(features.detach().float().std().cpu())
            return features, debug

        temporal = features
        relay = self.relay_tokens.to(device=features.device, dtype=features.dtype).expand(b, t, -1, -1)
        if self.relay_identity is not None:
            relay = relay + self.relay_identity.to(device=features.device, dtype=features.dtype)

        for layer_idx in range(self.num_layers):
            if self.local_enabled and not disable_local_neighborhood:
                temporal, local_debug = self.local_layers[layer_idx](temporal)
                debug.update(local_debug)
                debug[f"layer{layer_idx}_local_delta_ratio"] = local_debug["local_delta_ratio"]
            if self.global_enabled and not disable_global_relay:
                tokens = temporal.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
                tokens, relay, relay_debug = self.relay_layers[layer_idx](tokens, relay)
                temporal = tokens.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
                debug.update(relay_debug)
                debug[f"layer{layer_idx}_relay_delta_ratio"] = relay_debug["relay_delta_ratio"]

        spatial_tokens = features.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        temporal_tokens = temporal.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        fused = self.fusion_proj(torch.cat((self.spatial_norm(spatial_tokens), self.temporal_norm(temporal_tokens)), dim=-1))
        fused = fused.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        debug["temporal_output_std"] = float(fused.detach().float().std().cpu())
        return fused, debug
