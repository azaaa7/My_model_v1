from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoders import LiteBoundaryDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder
from .modules import NeighborhoodTemporalRelayFusion


class B23TemporalRelayLiteModel(nn.Module):
    """DINOv3-B23 + LoRA + neighborhood temporal relay + LiteBoundaryDecoder."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        dinov3_cfg = cfg.get("dinov3", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        decoder_cfg = dict(cfg.get("decoder", {}) or {})
        relay_cfg = dict(cfg.get("temporal_relay", {}) or {})
        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))
        self.logit_clamp = float((cfg.get("stability", {}) or {}).get("logit_clamp", 30.0))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        out_channels = int(decoder_cfg.get("in_channels", relay_cfg.get("dim", 256)))
        relay_cfg["dim"] = out_channels
        self.feature_proj = nn.Sequential(
            nn.GroupNorm(32, self.feature_dim),
            nn.Conv2d(self.feature_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.temporal_relay = NeighborhoodTemporalRelayFusion(relay_cfg)
        decoder_cfg["in_channels"] = out_channels
        self.decoder = LiteBoundaryDecoder(decoder_cfg)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.encoder_chunk and frames.shape[0] > self.encoder_chunk:
            chunks = []
            for part in frames.split(self.encoder_chunk, dim=0):
                if self.use_activation_checkpoint and self.training:
                    chunks.append(checkpoint(self.encoder, part, use_reentrant=False))
                else:
                    chunks.append(self.encoder(part))
            return torch.cat(chunks, dim=0)
        if self.use_activation_checkpoint and self.training:
            return checkpoint(self.encoder, frames, use_reentrant=False)
        return self.encoder(frames)

    @staticmethod
    def _restore(value: torch.Tensor | None, b: int, m: int, t: int):
        if value is None:
            return None
        return value.reshape(b, m, t, value.shape[-3], value.shape[-2], value.shape[-1])

    def forward(
        self,
        video: torch.Tensor,
        mode: str | None = None,
        ablation: dict[str, Any] | None = None,
        epoch: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del mode, epoch, kwargs
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,W,T,3,H,W], got {tuple(video.shape)}")
        ablation = ablation or {}
        b, m, t, c, h, w = video.shape
        frames = video.reshape(b * m * t, c, h, w)
        encoded = self.encode_frames(frames)
        _, feat_c, feat_h, feat_w = encoded.shape
        patch_features = encoded.reshape(b * m, t, feat_c, feat_h, feat_w)
        projected = self.feature_proj(encoded)
        _, dec_c, dec_h, dec_w = projected.shape
        projected_features = projected.reshape(b * m, t, dec_c, dec_h, dec_w)
        fused_features, temporal_debug = self.temporal_relay(
            projected_features,
            disable_temporal=bool(ablation.get("disable_temporal_relay", False)),
            disable_local_neighborhood=bool(ablation.get("disable_local_neighborhood", False)),
            disable_global_relay=bool(ablation.get("disable_global_relay", False)),
        )
        dec_out = self.decoder(fused_features.reshape(b * m * t, dec_c, dec_h, dec_w))
        logits = dec_out["logits"].reshape(b, m, t, 1, h, w)
        if self.logit_clamp > 0:
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        mask128 = self._restore(dec_out.get("mask128"), b, m, t)
        mask256 = self._restore(dec_out.get("mask256"), b, m, t)
        boundary128 = self._restore(dec_out.get("boundary128"), b, m, t)
        debug = {
            **temporal_debug,
            "patch_features_shape": tuple(patch_features.shape),
            "projected_features_shape": tuple(projected_features.shape),
            "fused_features_shape": tuple(fused_features.shape),
            "decoder_debug": dec_out.get("debug", {}),
        }
        return {
            "logits": logits,
            "aux": {
                "mask128": mask128,
                "mask256": mask256,
                "boundary128": boundary128,
                "edge_logits": boundary128,
                "ccm_mask32": None,
                "fgm_mask32": None,
                "fgm_cue": None,
                "debug": debug,
            },
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
