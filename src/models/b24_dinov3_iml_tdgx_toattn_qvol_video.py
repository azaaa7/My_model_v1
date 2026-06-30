from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .adapters import TemporalDifferenceGatedExcitation, TemporalOnlyAttention, TemporalTubeDropout
from .decoders import QueryVolumeDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder


class B24DINOv3IMLTDGXTOAttnQVolVideoModel(nn.Module):
    """DINOv3 + TubeDrop + TDGX + temporal-only attention + query volume decoder."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {}) or {}
        dinov3_cfg = (model_cfg.get("dinov3", {}) or cfg.get("dinov3", {}) or {})
        lora_cfg = (model_cfg.get("lora", {}) or cfg.get("lora", {}) or {})
        tube_cfg = (model_cfg.get("temporal_tube_dropout", {}) or cfg.get("temporal_tube_dropout", {}) or {})
        tdgx_cfg = (model_cfg.get("tdgx", {}) or cfg.get("tdgx", {}) or {})
        toattn_cfg = (model_cfg.get("temporal_only_attn", {}) or cfg.get("temporal_only_attn", {}) or {})
        qvol_cfg = dict((model_cfg.get("query_volume_decoder", {}) or cfg.get("query_volume_decoder", {}) or {}))

        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.input_size = int(cfg.get("input_size", dinov3_cfg.get("input_size", 512)))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.temporal_tube_dropout_enabled = bool(tube_cfg.get("enabled", True))
        self.temporal_tube_dropout = TemporalTubeDropout(float(tube_cfg.get("drop_prob", 0.10)))
        self.tdgx = TemporalDifferenceGatedExcitation(self.feature_dim, tdgx_cfg)
        self.temporal_only_attn = TemporalOnlyAttention(self.feature_dim, toattn_cfg)
        qvol_cfg.setdefault("in_channels", self.feature_dim)
        qvol_cfg.setdefault("image_size", self.input_size)
        qvol_cfg.setdefault("output_resolution", int(dinov3_cfg.get("output_resolution", 32)))
        qvol_cfg.setdefault("max_frames", int(cfg.get("num_frames", 4)))
        self.query_volume_decoder = QueryVolumeDecoder(qvol_cfg)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.encoder_chunk and frames.shape[0] > self.encoder_chunk:
            outputs = []
            for part in frames.split(self.encoder_chunk, dim=0):
                if self.use_activation_checkpoint and self.training:
                    outputs.append(checkpoint(self.encoder, part, use_reentrant=False))
                else:
                    outputs.append(self.encoder(part))
            return torch.cat(outputs, dim=0)
        if self.use_activation_checkpoint and self.training:
            return checkpoint(self.encoder, frames, use_reentrant=False)
        return self.encoder(frames)

    @staticmethod
    def _reshape_aux_list(items: list[torch.Tensor], b: int, m: int) -> list[torch.Tensor]:
        return [item.reshape(b, m, *item.shape[1:]) for item in items]

    def forward(
        self,
        video: torch.Tensor,
        mode: str | None = None,
        ablation: dict[str, Any] | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        del mode, epoch
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,M,K,3,H,W] or [B,K,3,H,W], got {tuple(video.shape)}")
        ablation = ablation or {}

        b, m, k, c, h_img, w_img = video.shape
        g = b * m
        flat_images = video.reshape(g * k, c, h_img, w_img)
        features_flat = self.encode_frames(flat_images)
        _, feat_c, feat_h, feat_w = features_flat.shape
        x = features_flat.reshape(g, k, feat_c, feat_h, feat_w)

        debug: dict[str, Any] = {
            "model": "B24DINOv3IMLTDGXTOAttnQVolVideoModel",
            "input_video_shape": tuple(video.shape),
            "feature_shape": tuple(x.shape),
        }

        if self.training and self.temporal_tube_dropout_enabled and not bool(ablation.get("disable_temporal_tube_dropout", False)):
            x = self.temporal_tube_dropout(x)
            debug["temporal_tube_dropout_enabled"] = True
        else:
            debug["temporal_tube_dropout_enabled"] = False

        x, tdgx_debug = self.tdgx(x)
        debug["tdgx"] = tdgx_debug
        x, attn_debug = self.temporal_only_attn(x)
        debug["temporal_only_attn"] = attn_debug

        dec_out = self.query_volume_decoder(x)
        logits = dec_out["logits"].reshape(b, m, k, 1, h_img, w_img)
        logits32 = dec_out["logits32"].reshape(b, m, k, 1, feat_h, feat_w)
        aux = {
            "logits32": logits32,
            "aux_logits": self._reshape_aux_list(dec_out.get("aux_logits", []), b, m),
            "aux_logits32": self._reshape_aux_list(dec_out.get("aux_logits32", []), b, m),
            "query_logits": dec_out.get("query_logits"),
            "query_scores": dec_out.get("query_scores"),
            "features32": x.reshape(b, m, *x.shape[1:]),
            "debug": {**debug, **dec_out.get("debug", {})},
        }
        return {"logits": logits, "aux": aux}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
