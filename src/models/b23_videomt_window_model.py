from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoders import LiteBoundaryDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder
from .videomt.query_encoder import VideoMTQueryController
from .videomt.query_mask_head import QueryMaskHead


class B23VideoMTWindowModel(nn.Module):
    """DINOv3-B23 with query tokens injected into final blocks and query masks."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        dinov3_cfg = dict(cfg.get("dinov3", {}) or {})
        lora_cfg = cfg.get("lora", {}) or {}
        videomt_cfg = cfg.get("videomt", {}) or {}
        decoder_cfg = dict(cfg.get("decoder", {}) or {})
        loss_cfg = cfg.get("loss", {}) or {}

        self.feature_dim = int(dinov3_cfg.get("feature_dim", videomt_cfg.get("dim", 1024)))
        self.out_channels = int(decoder_cfg.get("in_channels", decoder_cfg.get("out_channels", 128)))
        self.logit_clamp = float((cfg.get("stability", {}) or {}).get("logit_clamp", 30.0))
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        qinj_cfg = dict(dinov3_cfg.get("query_injection", {}) or {})
        qinj_cfg.setdefault("enabled", True)
        dinov3_cfg["query_injection"] = qinj_cfg
        dinov3_cfg.setdefault("use_activation_checkpoint", self.use_activation_checkpoint)

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.query_controller = VideoMTQueryController(videomt_cfg)
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

        stages = decoder_cfg.get("stages", []) or []
        f128_channels = int(stages[1].get("channels", 48)) if len(stages) > 1 else 48
        self.query_mask_head = QueryMaskHead(
            query_dim=self.feature_dim,
            in_mask_channels=f128_channels,
            cfg=videomt_cfg.get("query_mask_head", {}) or {},
        )

    def encode_frame_with_query(self, frame: torch.Tensor, q_in: torch.Tensor):
        if self.use_activation_checkpoint and self.training:
            return checkpoint(self.encoder.forward_with_queries, frame, q_in, use_reentrant=False)
        return self.encoder.forward_with_queries(frame, q_in)

    def forward(
        self,
        video: torch.Tensor,
        mode: str | None = None,
        ablation: dict[str, Any] | None = None,
        epoch: int | None = None,
        videomt_state: dict[str, torch.Tensor] | None = None,
        return_videomt_state: bool = False,
        **kwargs,
    ):
        del mode, epoch, kwargs
        ablation = ablation or {}
        if bool(ablation.get("disable_videomt", False)):
            raise ValueError("Final VidEoMT model does not support disable_videomt ablation.")
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,W,T,3,H,W], got {tuple(video.shape)}")

        batch, num_windows, num_frames, _channels, height, width = video.shape
        device = video.device
        prev_q = videomt_state.get("prev_q") if isinstance(videomt_state, dict) else None
        prop_cfg = ((self.cfg.get("videomt", {}) or {}).get("propagation", {}) or {})
        detach_across_windows = bool(prop_cfg.get("detach_across_windows", True))
        detach_within_window = bool(prop_cfg.get("detach_within_window", False))

        logits_per_window = []
        query_states_per_window = []
        query_logits_per_window = []
        query_scores_per_window = []
        edge_per_window = []
        debug: dict[str, Any] = {
            "input_video_shape": tuple(video.shape),
            "videomt_final": True,
            "query_injection": True,
            "videomt_enabled": True,
        }

        for win_idx in range(num_windows):
            patch_features_this_window = []
            queries_this_window = []
            for frame_idx in range(num_frames):
                frame = video[:, win_idx, frame_idx]
                detach_prev = bool(prev_q is not None and win_idx > 0 and frame_idx == 0 and detach_across_windows)
                q_in = self.query_controller.make_input_queries(
                    batch_size=batch,
                    device=device,
                    dtype=frame.dtype,
                    prev_q=prev_q,
                    detach_prev=detach_prev,
                )
                patch_feat, q_out = self.encode_frame_with_query(frame, q_in)
                patch_features_this_window.append(patch_feat)
                queries_this_window.append(q_out)
                prev_q = q_out.detach() if detach_within_window else q_out

            patch_features = torch.stack(patch_features_this_window, dim=1)
            query_states = torch.stack(queries_this_window, dim=1)
            feat_flat = patch_features.reshape(
                batch * num_frames,
                self.feature_dim,
                patch_features.shape[-2],
                patch_features.shape[-1],
            )
            dec_in = self.feature_proj(feat_flat)
            dec_out = self.decoder(dec_in)
            f128 = dec_out["features128"].reshape(
                batch,
                num_frames,
                dec_out["features128"].shape[1],
                dec_out["features128"].shape[-2],
                dec_out["features128"].shape[-1],
            )
            qmh_out = self.query_mask_head(
                query_states=query_states,
                mask_features=f128,
                output_size=(height, width),
            )
            logits = qmh_out["logits"]
            if self.logit_clamp > 0:
                logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
            logits_per_window.append(logits)
            query_states_per_window.append(query_states)
            query_logits_per_window.append(qmh_out["query_logits"])
            query_scores_per_window.append(qmh_out["query_scores"])

            edge_logits = dec_out.get("boundary128")
            if edge_logits is not None:
                edge_per_window.append(
                    edge_logits.reshape(batch, num_frames, 1, edge_logits.shape[-2], edge_logits.shape[-1])
                )

            debug[f"window{win_idx}_query"] = {
                "query_states_shape": tuple(query_states.shape),
                "query_logits_shape": tuple(qmh_out["query_logits"].shape),
                "query_scores_shape": tuple(qmh_out["query_scores"].shape),
            }
            debug[f"window{win_idx}_decoder"] = dec_out["debug"]

        logits = torch.stack(logits_per_window, dim=1)
        query_states = torch.stack(query_states_per_window, dim=1)
        query_logits = torch.stack(query_logits_per_window, dim=1)
        query_scores = torch.stack(query_scores_per_window, dim=1)
        aux = {
            "videomt_queries": query_states,
            "query_logits": query_logits,
            "query_scores": query_scores,
            "edge_logits": torch.stack(edge_per_window, dim=1) if edge_per_window else None,
            "boundary128": torch.stack(edge_per_window, dim=1) if edge_per_window else None,
            "ccm_mask32": None,
            "fgm_mask32": None,
            "fgm_cue": None,
            "debug": debug,
        }
        out = {"logits": logits, "aux": aux}
        if return_videomt_state:
            out["videomt_state"] = {"prev_q": prev_q.detach() if prev_q is not None else None}
        return out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
