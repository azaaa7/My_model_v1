from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoders import DINOv3IMLHead
from .dinov3_b23_encoder import DINOv3B23Encoder
from .tfcu import CCMLite


class B24DINOv3IMLCCMVideoModel(nn.Module):
    """DINOv3 B23 + optional CCM + DINOv3-IML 3-conv head for video."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        dinov3_cfg = cfg.get("dinov3", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        tfcu_cfg = cfg.get("tfcu", {}) or {}
        ccm_cfg = tfcu_cfg.get("ccm", {}) or {}
        decoder_cfg = dict(cfg.get("decoder", {}) or {})

        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.input_size = int(cfg.get("input_size", dinov3_cfg.get("input_size", 512)))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.ccm_enabled = bool(ccm_cfg.get("enabled", False))
        self.ccm = CCMLite(self.feature_dim, ccm_cfg)

        decoder_cfg.setdefault("type", "dinov3_iml_head")
        decoder_cfg.setdefault("in_channels", self.feature_dim)
        decoder_cfg.setdefault("hidden1", self.feature_dim // 2)
        decoder_cfg.setdefault("hidden2", self.feature_dim // 4)
        decoder_cfg.setdefault("image_size", self.input_size)
        decoder_cfg.setdefault("norm", "bn")
        self.decoder = DINOv3IMLHead(decoder_cfg)

        if not self.ccm_enabled:
            for param in self.ccm.parameters():
                param.requires_grad = False

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

    def _run_ccm(self, x_clip: torch.Tensor, disable_ccm: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if disable_ccm or not self.ccm_enabled:
            return x_clip, None, {
                "ccm_enabled": False,
                "ccm_disabled_by_ablation": bool(disable_ccm),
            }
        f_cc, _ccm_feat, ccm_aux, ccm_debug = self.ccm(x_clip)
        return f_cc, ccm_aux, ccm_debug

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
        disable_ccm = bool(ablation.get("disable_ccm", False))

        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)
        feat = self.encode_frames(frames)
        _, feat_c, feat_h, feat_w = feat.shape
        features = feat.reshape(b, m, k, feat_c, feat_h, feat_w)

        logits_list = []
        logits32_list = []
        ccm_aux_list = []
        debug: dict[str, Any] = {
            "model": "B24DINOv3IMLCCMVideoModel",
            "input_video_shape": tuple(video.shape),
            "feature_shape": tuple(features.shape),
            "ccm_enabled": self.ccm_enabled,
        }

        for clip_idx in range(m):
            x_clip = features[:, clip_idx]
            x_dec, ccm_aux, ccm_debug = self._run_ccm(x_clip, disable_ccm=disable_ccm)
            dec = self.decoder(x_dec.reshape(b * k, feat_c, feat_h, feat_w), output_size=(h, w))
            logits_list.append(dec["logits"].reshape(b, k, 1, h, w))
            logits32_list.append(dec["logits32"].reshape(b, k, 1, feat_h, feat_w))
            if ccm_aux is not None:
                ccm_aux_list.append(ccm_aux)
            debug[f"clip{clip_idx}_ccm"] = ccm_debug
            debug[f"clip{clip_idx}_decoder"] = dec["debug"]

        logits = torch.stack(logits_list, dim=1)
        logits32 = torch.stack(logits32_list, dim=1)
        aux = {
            "logits32": logits32,
            "ccm_mask32": torch.stack(ccm_aux_list, dim=1) if ccm_aux_list else None,
            "debug": debug,
        }
        return {"logits": logits, "aux": aux}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
