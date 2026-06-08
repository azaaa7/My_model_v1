from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decoders import LiteBoundaryDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder
from .forensic_branch import ResidualNoiseForensicBranch
from .tfcu import CCMLite, FGMLite, LowResTFCUFusion, StaticLowResFusion


class B23TFCUCCMFGMLiteModel(nn.Module):
    """DINOv3-B23 + LoRA32 + CCM-Lite + FGM-Lite + low-res decoder."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        dinov3_cfg = cfg.get("dinov3", {})
        lora_cfg = cfg.get("lora", {})
        tfcu_cfg = cfg.get("tfcu", {})
        fusion_cfg = tfcu_cfg.get("fusion", {}) or cfg.get("fusion", {})
        forensic_cfg = cfg.get("forensic_branch", {}) or {}
        decoder_cfg = cfg.get("decoder", {})
        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))
        self.tfcu_version = str(tfcu_cfg.get("version", "ccm_fgm_lite"))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.ccm = CCMLite(self.feature_dim, tfcu_cfg.get("ccm", {}))
        self.fgm = FGMLite(self.feature_dim, tfcu_cfg.get("fgm", {}))
        ccm_enabled = bool((tfcu_cfg.get("ccm", {}) or {}).get("enabled", True))
        fgm_enabled = bool((tfcu_cfg.get("fgm", {}) or {}).get("enabled", True))
        self.use_temporal = ccm_enabled or fgm_enabled
        self.static_adapter = StaticLowResFusion(self.feature_dim, int(fusion_cfg.get("out_channels", 128)))
        self.fusion = LowResTFCUFusion(self.feature_dim, fusion_cfg)
        forensic_cfg = dict(forensic_cfg)
        forensic_cfg.setdefault("fusion_channels", int(fusion_cfg.get("out_channels", 128)))
        forensic_cfg.setdefault("target_resolution", int(fusion_cfg.get("q_resolution", 32)))
        self.forensic_branch = ResidualNoiseForensicBranch(forensic_cfg)
        self.forensic_enabled = bool(forensic_cfg.get("enabled", False))
        self.bank_quality_cfg = cfg.get("fgm_bank", {}) or {}
        self.decoder = LiteBoundaryDecoder(decoder_cfg)
        if not self.forensic_enabled:
            for param in self.forensic_branch.parameters():
                param.requires_grad = False
        if not ccm_enabled:
            for param in self.ccm.parameters():
                param.requires_grad = False
        if not fgm_enabled:
            for param in self.fgm.parameters():
                param.requires_grad = False
        if self.use_temporal:
            for param in self.static_adapter.parameters():
                param.requires_grad = False
        else:
            for param in self.fusion.parameters():
                param.requires_grad = False

    def _quality_gate_cue(self, cue: torch.Tensor, logits_clip: torch.Tensor, bank) -> tuple[torch.Tensor, bool, dict[str, float]]:
        gate_cfg = self.bank_quality_cfg.get("quality_gate", {}) or {}
        if not bool(gate_cfg.get("enabled", False)):
            return cue, True, {"fgm_bank_quality_enabled": False}

        prob = torch.sigmoid(logits_clip.detach().float())
        eps = 1.0e-6
        entropy = -(prob * torch.log(prob.clamp_min(eps)) + (1.0 - prob) * torch.log((1.0 - prob).clamp_min(eps)))
        confidence = 1.0 - entropy.mean() / 0.69314718056
        area = prob.mean()
        area_min = float(gate_cfg.get("area_min", 0.001))
        area_max = float(gate_cfg.get("area_max", 0.70))
        if area <= area_min:
            area_score = area / max(area_min, eps)
        elif area >= area_max:
            area_score = (1.0 - area) / max(1.0 - area_max, eps)
        else:
            area_score = torch.ones_like(area)
        area_score = area_score.clamp(0.0, 1.0)

        consistency = torch.ones_like(confidence)
        prev = bank.last() if hasattr(bank, "last") else None
        if prev is not None:
            cue_vec = F.normalize(cue.detach().float().flatten(1), dim=1)
            prev_vec = F.normalize(prev.detach().float().flatten(1), dim=1)
            similarity = (cue_vec * prev_vec).sum(dim=1).mean()
            min_sim = float(gate_cfg.get("min_similarity", 0.05))
            consistency = ((similarity - min_sim) / max(1.0 - min_sim, eps)).clamp(0.0, 1.0)

        conf_weight = float(gate_cfg.get("confidence_weight", 0.45))
        area_weight = float(gate_cfg.get("area_weight", 0.25))
        consistency_weight = float(gate_cfg.get("consistency_weight", 0.30))
        total_weight = max(conf_weight + area_weight + consistency_weight, eps)
        quality = (
            conf_weight * confidence.clamp(0.0, 1.0)
            + area_weight * area_score
            + consistency_weight * consistency
        ) / total_weight
        warmup = int(gate_cfg.get("warmup_items", 1))
        keep_threshold = float(gate_cfg.get("keep_threshold", 0.20))
        scale_min = float(gate_cfg.get("scale_min", 0.25))
        scale = scale_min + (1.0 - scale_min) * quality.clamp(0.0, 1.0)
        should_keep = len(bank) < warmup or bool((quality >= keep_threshold).detach().cpu())
        gated_cue = cue * scale.to(device=cue.device, dtype=cue.dtype)
        debug = {
            "fgm_bank_quality_enabled": True,
            "fgm_bank_quality": float(quality.detach().cpu()),
            "fgm_bank_quality_confidence": float(confidence.detach().cpu()),
            "fgm_bank_quality_area": float(area.detach().cpu()),
            "fgm_bank_quality_area_score": float(area_score.detach().cpu()),
            "fgm_bank_quality_consistency": float(consistency.detach().cpu()),
            "fgm_bank_quality_scale": float(scale.detach().cpu()),
            "fgm_bank_quality_keep": float(1.0 if should_keep else 0.0),
        }
        return gated_cue, should_keep, debug

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

    def _maybe_checkpoint_ccm(self, x_m: torch.Tensor):
        if self.use_activation_checkpoint and self.training:
            return checkpoint(lambda y: self.ccm(y)[:3], x_m, use_reentrant=False)
        return self.ccm(x_m)[:3]

    def new_fgm_bank(self, ablation: dict[str, Any] | None = None, detach_bank: bool | None = None):
        ablation = ablation or {}
        bank_cfg = self.cfg.get("fgm_bank", {}) or {}
        if detach_bank is None and "detach_cross_window" in bank_cfg:
            detach_bank = bool(bank_cfg.get("detach_cross_window", True))
        return self.fgm.new_bank(
            shuffle_bank=bool(ablation.get("shuffle_bank", False)),
            zero_bank=bool(ablation.get("zero_bank", False)),
            detach_bank=detach_bank,
        )

    def forward(
        self,
        video: torch.Tensor,
        mode: str | None = None,
        ablation: dict[str, Any] | None = None,
        fgm_bank=None,
        return_fgm_bank: bool = False,
    ):
        """Forward [B,M,K,3,512,512] -> dict with logits [B,M,K,1,512,512]."""
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,M,K,3,H,W], got {tuple(video.shape)}")
        ablation = ablation or {}
        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)
        b23 = self.encode_frames(frames)
        forensic_feat, forensic_debug = self.forensic_branch(frames)
        _, forensic_c, forensic_h, forensic_w = forensic_feat.shape
        forensic_feat = forensic_feat.reshape(b, m, k, forensic_c, forensic_h, forensic_w)
        _, feat_c, feat_h, feat_w = b23.shape
        features = b23.reshape(b, m, k, feat_c, feat_h, feat_w)

        bank = fgm_bank if fgm_bank is not None else self.new_fgm_bank(ablation)
        disable_ccm = bool(ablation.get("disable_ccm", False))
        disable_fgm = bool(ablation.get("disable_fgm", False))

        logits_list = []
        mask128_list = []
        mask256_list = []
        boundary128_list = []
        ccm_aux_list = []
        fgm_aux_list = []
        cue_list = []
        debug: dict[str, Any] = {
            "input_video_shape": tuple(video.shape),
            "b23_feature_shape": tuple(b23.shape),
            "forensic_branch": forensic_debug,
        }

        for clip_idx in range(m):
            x_m = features[:, clip_idx]
            if disable_ccm:
                f_cc = x_m
                ccm_feat = torch.zeros(b, k, self.ccm.dim, feat_h, feat_w, device=x_m.device, dtype=x_m.dtype)
                ccm_aux = None
                ccm_debug = {"ccm_enabled": False, "ccm_disabled_by_ablation": True}
            else:
                if self.use_activation_checkpoint and self.training:
                    f_cc, ccm_feat, ccm_aux = self._maybe_checkpoint_ccm(x_m)
                    with torch.no_grad():
                        _, _, _, ccm_debug = self.ccm(x_m.detach())
                else:
                    f_cc, ccm_feat, ccm_aux, ccm_debug = self.ccm(x_m)

            if disable_fgm:
                f_ip = torch.zeros_like(x_m)
                cue = self.fgm._current_key(f_cc)
                fgm_aux = None
                fgm_debug = {"fgm_enabled": False, "fgm_disabled_by_ablation": True, "fgm_bank_size": len(bank)}
            else:
                f_ip, cue, fgm_aux, fgm_debug = self.fgm(x_m, f_cc, bank)

            if self.use_temporal and not (disable_ccm and disable_fgm):
                f32, fusion_debug = self.fusion(x_m, f_cc, f_ip)
            else:
                f32, fusion_debug = self.static_adapter(x_m)
            if self.forensic_enabled:
                forensic_clip = forensic_feat[:, clip_idx].reshape(b * k, forensic_c, forensic_h, forensic_w)
                if forensic_clip.shape[-2:] != f32.shape[-2:]:
                    forensic_clip = F.interpolate(forensic_clip, size=f32.shape[-2:], mode="bilinear", align_corners=False)
                f32 = f32 + forensic_clip
                fusion_debug["fusion_forensic_norm"] = float(forensic_clip.detach().float().norm().cpu())

            dec = self.decoder(f32)
            logits_clip = dec["logits"].reshape(b, k, 1, h, w)
            mask128_clip = dec["mask128"].reshape(b, k, 1, 128, 128)
            mask256_clip = dec["mask256"].reshape(b, k, 1, 256, 256)
            boundary128 = dec["boundary128"]
            if boundary128 is not None:
                boundary128 = boundary128.reshape(b, k, 1, 128, 128)

            logits_list.append(logits_clip)
            mask128_list.append(mask128_clip)
            mask256_list.append(mask256_clip)
            boundary128_list.append(boundary128)
            if ccm_aux is not None:
                ccm_aux_list.append(ccm_aux)
            if fgm_aux is not None:
                fgm_aux_list.append(fgm_aux)
            cue_list.append(cue)
            if not disable_fgm:
                gated_cue, keep_cue, bank_debug = self._quality_gate_cue(cue, logits_clip, bank)
                fgm_debug.update(bank_debug)
                if keep_cue:
                    bank.append(gated_cue)

            debug[f"clip{clip_idx}_ccm"] = ccm_debug
            debug[f"clip{clip_idx}_fgm"] = fgm_debug
            debug[f"clip{clip_idx}_fusion"] = fusion_debug
            debug[f"clip{clip_idx}_decoder"] = dec["debug"]

        logits = torch.stack(logits_list, dim=1)
        aux = {
            "mask128": torch.stack(mask128_list, dim=1),
            "mask256": torch.stack(mask256_list, dim=1),
            "boundary128": torch.stack(boundary128_list, dim=1) if all(x is not None for x in boundary128_list) else None,
            "ccm_mask32": torch.stack(ccm_aux_list, dim=1) if ccm_aux_list else None,
            "fgm_mask32": torch.stack(fgm_aux_list, dim=1) if fgm_aux_list else None,
            "fgm_cue": torch.stack(cue_list, dim=1),
            "debug": debug,
        }
        out = {"logits": logits, "aux": aux}
        if return_fgm_bank:
            out["fgm_bank"] = bank
        return out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_trainable_by_keyword(model: nn.Module, keywords: tuple[str, ...]) -> int:
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad and any(key in name for key in keywords):
            total += param.numel()
    return total
