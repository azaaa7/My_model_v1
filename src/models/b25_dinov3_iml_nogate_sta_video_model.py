from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .adapters import NoGateSpatiotemporalAdapter
from .decoders import BoundedResidualMicroRefineDecoder, DINOv3IMLHead
from .dinov3_b23_encoder import DINOv3B23Encoder


class B25DINOv3IMLNoGateStAVideoModel(nn.Module):
    """DINOv3 B23 + NoGate StA + DINOv3-IML 3-conv head for video."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        dinov3_cfg = cfg.get("dinov3", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        adapter_cfg = dict(cfg.get("nogate_sta", {}) or {})
        decoder_cfg = dict(cfg.get("decoder", {}) or {})

        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.input_size = int(cfg.get("input_size", dinov3_cfg.get("input_size", 512)))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        adapter_cfg.setdefault("enabled", True)
        self.nogate_sta = NoGateSpatiotemporalAdapter(self.feature_dim, adapter_cfg)

        decoder_cfg.setdefault("type", "dinov3_iml_head")
        decoder_cfg.setdefault("in_channels", self.feature_dim)
        decoder_cfg.setdefault("hidden1", self.feature_dim // 2)
        decoder_cfg.setdefault("hidden2", self.feature_dim // 4)
        decoder_cfg.setdefault("image_size", self.input_size)
        decoder_cfg.setdefault("norm", "bn")
        decoder_type = str(decoder_cfg.get("type", "dinov3_iml_head")).lower()
        if decoder_type in {"bounded_residual_micro_refine", "brmr"}:
            self.decoder = BoundedResidualMicroRefineDecoder(decoder_cfg)
        else:
            self.decoder = DINOv3IMLHead(decoder_cfg)

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
        disable_sta = bool(ablation.get("disable_sta", False))

        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)
        feat = self.encode_frames(frames)
        _, feat_c, feat_h, feat_w = feat.shape
        features = feat.reshape(b, m, k, feat_c, feat_h, feat_w)

        logits_list = []
        logits32_list = []
        logits32_coarse_list = []
        logits128_list = []
        delta32_list = []
        delta128_list = []
        debug: dict[str, Any] = {
            "model": "B25DINOv3IMLNoGateStAVideoModel",
            "input_video_shape": tuple(video.shape),
            "feature_shape": tuple(features.shape),
            "disable_sta": disable_sta,
        }

        for clip_idx in range(m):
            x_clip = features[:, clip_idx]
            if disable_sta:
                x_adapt = x_clip
                sta_debug = {"nogate_sta_enabled": False, "disabled_by_ablation": True}
            else:
                x_adapt, sta_debug = self.nogate_sta(x_clip)
            if hasattr(self.decoder, "forward_video"):
                dec = self.decoder.forward_video(x_adapt, output_size=(h, w))
                logits_list.append(dec["logits"])
                logits32_list.append(dec["logits32"])
                if "logits32_coarse" in dec:
                    logits32_coarse_list.append(dec["logits32_coarse"])
                if "logits128" in dec:
                    logits128_list.append(dec["logits128"])
                if "delta32" in dec:
                    delta32_list.append(dec["delta32"])
                if "delta128" in dec:
                    delta128_list.append(dec["delta128"])
            else:
                dec = self.decoder(x_adapt.reshape(b * k, feat_c, feat_h, feat_w), output_size=(h, w))
                logits_list.append(dec["logits"].reshape(b, k, 1, h, w))
                logits32_list.append(dec["logits32"].reshape(b, k, 1, feat_h, feat_w))
            debug[f"clip{clip_idx}_nogate_sta"] = sta_debug
            debug[f"clip{clip_idx}_decoder"] = dec["debug"]

        logits = torch.stack(logits_list, dim=1)
        logits32 = torch.stack(logits32_list, dim=1)
        aux: dict[str, Any] = {
            "logits32": logits32,
            "debug": debug,
        }
        if logits32_coarse_list:
            aux["logits32_coarse"] = torch.stack(logits32_coarse_list, dim=1)
        if logits128_list:
            aux["logits128"] = torch.stack(logits128_list, dim=1)
        if delta32_list:
            aux["delta32"] = torch.stack(delta32_list, dim=1)
        if delta128_list:
            aux["delta128"] = torch.stack(delta128_list, dim=1)
        return {
            "logits": logits,
            "aux": aux,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
