from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoders import LiteBoundaryDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder
from .videomt import WindowQueryFusion


class B23VideoMTWindowModel(nn.Module):
    """DINOv3-B23 + VidEoMT-style window query fusion + lite decoder."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        dinov3_cfg = cfg.get("dinov3", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        videomt_cfg = cfg.get("videomt", {}) or {}
        decoder_cfg = dict(cfg.get("decoder", {}) or {})
        loss_cfg = cfg.get("loss", {}) or {}

        self.feature_dim = int(dinov3_cfg.get("feature_dim", videomt_cfg.get("dim", 1024)))
        self.out_channels = int(decoder_cfg.get("in_channels", decoder_cfg.get("out_channels", 128)))
        self.logit_clamp = float((cfg.get("stability", {}) or {}).get("logit_clamp", 30.0))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.query_fusion = WindowQueryFusion(videomt_cfg)
        self.feature_proj = nn.Sequential(
            nn.GroupNorm(32, self.feature_dim),
            nn.Conv2d(self.feature_dim, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.out_channels),
            nn.GELU(),
        )
        decoder_cfg["in_channels"] = self.out_channels
        edge_weight = float(loss_cfg.get("edge_weight", 0.0) or 0.0)
        mask128_cfg = dict(decoder_cfg.get("mask128_head", {}) or {})
        mask128_cfg["enabled"] = bool(mask128_cfg.get("enabled", False))
        decoder_cfg["mask128_head"] = mask128_cfg
        boundary_cfg = dict(decoder_cfg.get("boundary_head", {}) or {})
        boundary_cfg["enabled"] = bool(boundary_cfg.get("enabled", False) and edge_weight > 0.0)
        decoder_cfg["boundary_head"] = boundary_cfg
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

    def forward(self, video: torch.Tensor, mode: str | None = None, ablation: dict[str, Any] | None = None, epoch: int | None = None):
        del mode, epoch
        ablation = ablation or {}
        if bool(ablation.get("disable_videomt", False)):
            raise ValueError("B23VideoMTWindowModel does not support disable_videomt ablation in the final model.")
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,W,T,3,H,W], got {tuple(video.shape)}")

        batch, num_windows, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch * num_windows * num_frames, channels, height, width)
        b23 = self.encode_frames(frames)
        _, feat_c, feat_h, feat_w = b23.shape
        features = b23.reshape(batch, num_windows, num_frames, feat_c, feat_h, feat_w)

        logits_per_window = []
        query_states_per_window = []
        edge_per_window = []
        debug: dict[str, Any] = {
            "input_video_shape": tuple(video.shape),
            "b23_feature_shape": tuple(b23.shape),
            "videomt_enabled": True,
        }

        for win_idx in range(num_windows):
            x_win = features[:, win_idx]
            enhanced_win, query_aux = self.query_fusion(x_win)
            enhanced_flat = enhanced_win.reshape(batch * num_frames, feat_c, feat_h, feat_w)
            dec_in = self.feature_proj(enhanced_flat)
            dec_out = self.decoder(dec_in)

            logits = dec_out["logits"].reshape(batch, num_frames, 1, height, width)
            if self.logit_clamp > 0:
                logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
            logits_per_window.append(logits)
            query_states_per_window.append(query_aux["query_states"])

            edge_logits = dec_out.get("edge_logits")
            if edge_logits is None:
                edge_logits = dec_out.get("boundary128")
            if edge_logits is not None:
                edge_per_window.append(
                    edge_logits.reshape(batch, num_frames, 1, edge_logits.shape[-2], edge_logits.shape[-1])
                )

            debug[f"window{win_idx}_videomt"] = {
                "videomt_query_alpha": query_aux["videomt_query_alpha"],
                "videomt_query_shape": query_aux["videomt_query_shape"],
            }
            debug[f"window{win_idx}_decoder"] = dec_out["debug"]

        logits = torch.stack(logits_per_window, dim=1)
        query_states = torch.stack(query_states_per_window, dim=1)
        aux = {
            "videomt_queries": query_states,
            "edge_logits": torch.stack(edge_per_window, dim=1) if edge_per_window else None,
            "boundary128": torch.stack(edge_per_window, dim=1) if edge_per_window else None,
            "ccm_mask32": None,
            "fgm_mask32": None,
            "fgm_cue": None,
            "debug": debug,
        }
        out = {"logits": logits, "aux": aux}
        return out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
