from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decoders import LiteBoundaryDecoder
from .dinov3_b23_encoder import DINOv3B23Encoder
from .forensic_branch import ResidualNoiseForensicBranch
from .modules import TaskSpecificForensicsAdapter, TemporalCueUnraveling
from .tfcu import (
    CCMLite,
    FGMLite,
    HP3DNoiseAdapter,
    LowResTFCUFusion,
    PrototypeMemory,
    ReliabilityGate,
    StaticLowResFusion,
    binary_entropy,
)
from src.losses.sumi_localization_losses import SUMIMinimalityHeads


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
        reliability_cfg = cfg.get("reliability_gate", {}) or {}
        forensic_adapter_cfg = cfg.get("forensic_adapter", {}) or cfg.get("hp3d_noise_adapter", {}) or {}
        prototype_cfg = cfg.get("prototype_memory", {}) or {}
        task_adapter_cfg = cfg.get("task_forensics_adapter", {}) or {}
        tcu_cfg = cfg.get("temporal_cue_unraveling", {}) or {}
        sumi_cfg = ((cfg.get("loss", {}) or {}).get("sumi", {}) or cfg.get("sumi", {}) or {})
        self.sumi_cfg = sumi_cfg
        stability_cfg = cfg.get("stability", {}) or {}
        decoder_cfg = cfg.get("decoder", {})
        temporal_encoder_cfg = dict(cfg.get("temporal_encoder", {}) or {})
        temporal_encoder_cfg.setdefault("use_activation_checkpoint", bool(cfg.get("use_activation_checkpoint", False)))
        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))
        self.use_temporal_encoder = bool(temporal_encoder_cfg.get("enabled", False))
        self.tfcu_version = str(tfcu_cfg.get("version", "ccm_fgm_lite"))
        self.fusion_channels = int(fusion_cfg.get("out_channels", 128))
        self.fgm_feedback_max = float(stability_cfg.get("fgm_feedback_max", 0.05))
        self.logit_clamp = float(stability_cfg.get("logit_clamp", 30.0))
        self.ccm_alpha_max = float((tfcu_cfg.get("ccm", {}) or {}).get("alpha_max", 0.0))

        if self.use_temporal_encoder:
            from .dinov3_b23_temporal_encoder import DINOv3B23TemporalEncoder

            self.encoder = DINOv3B23TemporalEncoder(
                dinov3_cfg=dinov3_cfg,
                lora_cfg=lora_cfg,
                temporal_cfg=temporal_encoder_cfg,
            )
        else:
            self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
        self.task_adapter = TaskSpecificForensicsAdapter(self.feature_dim, task_adapter_cfg)
        self.task_adapter_enabled = bool(task_adapter_cfg.get("enabled", False))
        self.tcu = TemporalCueUnraveling(self.feature_dim, tcu_cfg)
        self.tcu_enabled = bool(tcu_cfg.get("enabled", False))
        self.ccm = CCMLite(self.feature_dim, tfcu_cfg.get("ccm", {}))
        self.fgm = FGMLite(self.feature_dim, tfcu_cfg.get("fgm", {}))
        ccm_enabled = bool((tfcu_cfg.get("ccm", {}) or {}).get("enabled", True))
        fgm_enabled = bool((tfcu_cfg.get("fgm", {}) or {}).get("enabled", True))
        self.ccm_enabled = ccm_enabled
        self.fgm_enabled = fgm_enabled
        self.use_temporal = ccm_enabled or fgm_enabled or self.tcu_enabled
        self.static_adapter = StaticLowResFusion(self.feature_dim, self.fusion_channels)
        self.fusion = LowResTFCUFusion(self.feature_dim, fusion_cfg)
        forensic_cfg = dict(forensic_cfg)
        forensic_cfg.setdefault("fusion_channels", self.fusion_channels)
        forensic_cfg.setdefault("target_resolution", int(fusion_cfg.get("q_resolution", 32)))
        self.forensic_branch = ResidualNoiseForensicBranch(forensic_cfg)
        self.forensic_enabled = bool(forensic_cfg.get("enabled", False))
        forensic_adapter_cfg = dict(forensic_adapter_cfg)
        forensic_adapter_cfg.setdefault("target_resolution", int(fusion_cfg.get("q_resolution", 32)))
        self.noise_adapter = HP3DNoiseAdapter(
            forensic_adapter_cfg,
            out_channels=self.fusion_channels,
            target_resolution=int(fusion_cfg.get("q_resolution", 32)),
        )
        self.noise_adapter_enabled = bool(forensic_adapter_cfg.get("enabled", False))
        self.reliability_gate = ReliabilityGate(reliability_cfg, in_channels=4)
        self.reliability_gate_enabled = bool(reliability_cfg.get("enabled", False))
        self.prototype_memory = PrototypeMemory(prototype_cfg, channels=self.fusion_channels)
        self.prototype_memory_enabled = bool(prototype_cfg.get("enabled", False))
        self.prototype_replace_raw_bank = bool(prototype_cfg.get("replace_raw_bank", False))
        self.sumi_heads = SUMIMinimalityHeads(
            channels=self.fusion_channels,
            bottleneck_dim=int((sumi_cfg.get("minimality", {}) or {}).get("bottleneck_dim", 64)),
            num_sources=int((sumi_cfg.get("source_adversarial", {}) or {}).get("num_sources", 3)),
        )
        self.sumi_enabled = bool(sumi_cfg.get("enabled", False))
        self.bank_quality_cfg = cfg.get("fgm_bank", {}) or {}
        self.decoder = LiteBoundaryDecoder(decoder_cfg)
        if not self.task_adapter_enabled:
            for param in self.task_adapter.parameters():
                param.requires_grad = False
        if not self.tcu_enabled:
            for param in self.tcu.parameters():
                param.requires_grad = False
        if not self.sumi_enabled:
            for param in self.sumi_heads.parameters():
                param.requires_grad = False
        if not self.forensic_enabled:
            for param in self.forensic_branch.parameters():
                param.requires_grad = False
        if not self.noise_adapter_enabled:
            for param in self.noise_adapter.parameters():
                param.requires_grad = False
        if not self.reliability_gate_enabled:
            for param in self.reliability_gate.parameters():
                param.requires_grad = False
        if not self.prototype_memory_enabled:
            for param in self.prototype_memory.parameters():
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

    def _clamp_trainable_scalars(self) -> None:
        if self.ccm_alpha_max > 0 and hasattr(self.ccm, "alpha_cc"):
            with torch.no_grad():
                self.ccm.alpha_cc.clamp_(-self.ccm_alpha_max, self.ccm_alpha_max)
        if self.fgm_feedback_max > 0 and hasattr(self.fgm, "cue_feedback_scale"):
            with torch.no_grad():
                self.fgm.cue_feedback_scale.clamp_(-self.fgm_feedback_max, self.fgm_feedback_max)

    @staticmethod
    def _fgm_gate_inputs(
        fgm_aux: torch.Tensor | None,
        noise_reliability: torch.Tensor | None,
        b: int,
        k: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if fgm_aux is not None:
            prob = torch.sigmoid(fgm_aux.reshape(b * k, 1, fgm_aux.shape[-2], fgm_aux.shape[-1]).float())
            if prob.shape[-2:] != (h, w):
                prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
            mask_entropy = binary_entropy(prob).to(dtype=dtype)
            mask_conf = (1.0 - mask_entropy).to(dtype=dtype)
        else:
            mask_entropy = torch.zeros(b * k, 1, h, w, device=device, dtype=dtype)
            mask_conf = torch.full_like(mask_entropy, 0.5)
        if noise_reliability is None:
            noise = torch.full_like(mask_entropy, 0.5)
        else:
            noise = noise_reliability
            if noise.shape[-2:] != (h, w):
                noise = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
            noise = noise.to(device=device, dtype=dtype)
        motion = torch.zeros_like(mask_entropy)
        return [mask_entropy, mask_conf, noise, motion]

    @staticmethod
    def _mean_debug(debug: dict[str, Any], prefix: str) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for key, item in debug.items():
            if not str(key).startswith("clip") or not isinstance(item, dict):
                continue
            for sub_key, sub_value in item.items():
                if isinstance(sub_value, (int, float)) and str(sub_key).startswith(prefix):
                    values.setdefault(sub_key, []).append(float(sub_value))
        return {key: sum(items) / max(len(items), 1) for key, items in values.items()}

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

    def encode_video_or_frames(self, video: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        b, m, k, c, h, w = video.shape
        if self.use_temporal_encoder:
            feat, debug, temporal_aux = self.encoder(video, return_aux=True)
            return feat, debug, temporal_aux
        frames = video.reshape(b * m * k, c, h, w)
        feat = self.encode_frames(frames)
        debug = {
            "temporal_encoder_enabled": False,
            "temporal_encoder_type": "DINOv3B23Encoder",
            "patch_hw": tuple(feat.shape[-2:]),
            "feat_shape": tuple(feat.shape),
        }
        return feat, debug, {}

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
        epoch: int | None = None,
    ):
        """Forward [B,M,K,3,512,512] -> dict with logits [B,M,K,1,512,512]."""
        if video.ndim == 5:
            video = video[:, None]
        if video.ndim != 6:
            raise ValueError(f"video must be [B,M,K,3,H,W], got {tuple(video.shape)}")
        ablation = ablation or {}
        self._clamp_trainable_scalars()
        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)
        b23, temporal_debug, temporal_aux = self.encode_video_or_frames(video)
        forensic_feat, forensic_debug = self.forensic_branch(frames)
        _, forensic_c, forensic_h, forensic_w = forensic_feat.shape
        forensic_feat = forensic_feat.reshape(b, m, k, forensic_c, forensic_h, forensic_w)
        _, feat_c, feat_h, feat_w = b23.shape
        features = b23.reshape(b, m, k, feat_c, feat_h, feat_w)
        adapter_aux_list = []
        if self.task_adapter_enabled and not bool(ablation.get("disable_task_adapter", False)):
            adapted = []
            for clip_idx in range(m):
                feat_clip, adapter_aux = self.task_adapter(features[:, clip_idx], video[:, clip_idx], epoch=epoch)
                adapted.append(feat_clip)
                adapter_aux_list.append(adapter_aux)
            features = torch.stack(adapted, dim=1)

        bank = fgm_bank if fgm_bank is not None else self.new_fgm_bank(ablation)
        disable_ccm = bool(ablation.get("disable_ccm", False)) or not self.ccm_enabled
        disable_fgm = bool(ablation.get("disable_fgm", False)) or not self.fgm_enabled

        logits_list = []
        mask128_list = []
        mask256_list = []
        boundary128_list = []
        ccm_aux_list = []
        fgm_aux_list = []
        adapter_mask32_list = []
        adapter_boundary32_list = []
        adapter_gate_list = []
        tcu_momentary_list = []
        tcu_gradual_list = []
        tcu_cumulative_list = []
        tcu_logit_delta_list = []
        sumi_ib_kl_list = []
        sumi_source_logits_list = []
        sumi_activation32_list = []
        noise_mask32_list = []
        noise_boundary32_list = []
        gate_regularization_list = []
        cue_list = []
        debug: dict[str, Any] = {
            "input_video_shape": tuple(video.shape),
            "b23_feature_shape": tuple(b23.shape),
            "temporal_encoder": temporal_debug,
            "forensic_branch": forensic_debug,
        }
        if adapter_aux_list:
            for aux_item in adapter_aux_list:
                adapter_mask32_list.append(aux_item["adapter_mask32"])
                adapter_boundary32_list.append(aux_item["adapter_boundary32"])
                adapter_gate_list.append(aux_item["adapter_gate"])
            debug["task_forensics_adapter"] = {
                key: sum(float((item.get("debug", {}) or {}).get(key, 0.0)) for item in adapter_aux_list) / max(len(adapter_aux_list), 1)
                for key in ("adapter_alpha", "adapter_gate_mean", "adapter_delta_norm")
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

            tcu_active = False
            if self.tcu_enabled and not bool(ablation.get("disable_tcu", False)):
                tcu_aux = self.tcu(f_cc, lowres_logits=ccm_aux, rgb=video[:, clip_idx])
                tcu_zero_dep = (
                    tcu_aux["logit_delta32"].sum()
                    + tcu_aux["tcu_momentary_mask32"].sum()
                    + tcu_aux["tcu_gradual_mask32"].sum()
                    + tcu_aux["tcu_cumulative_mask32"].sum()
                ) * 0.0
                f_cc = f_cc + tcu_aux["feat_delta"] + tcu_zero_dep
                tcu_active = True
                tcu_momentary_list.append(tcu_aux["tcu_momentary_mask32"])
                tcu_gradual_list.append(tcu_aux["tcu_gradual_mask32"])
                tcu_cumulative_list.append(tcu_aux["tcu_cumulative_mask32"])
                tcu_logit_delta_list.append(tcu_aux["logit_delta32"])
                ccm_debug.update(tcu_aux.get("debug", {}))

            if disable_fgm:
                f_ip = torch.zeros_like(x_m)
                cue = self.fgm._current_key(f_cc)
                fgm_aux = None
                fgm_debug = {"fgm_enabled": False, "fgm_disabled_by_ablation": True, "fgm_bank_size": len(bank)}
            else:
                f_ip, cue, fgm_aux, fgm_debug = self.fgm(x_m, f_cc, bank)

            noise_feat = None
            noise_reliability = None
            noise_aux = None
            if self.noise_adapter_enabled and not bool(ablation.get("disable_noise_adapter", False)):
                noise_frames = video[:, clip_idx]
                noise_feat, noise_reliability, noise_aux = self.noise_adapter.extract_features(noise_frames, x_m.dtype)
                noise_mask32_list.append(noise_aux["noise_mask32"])
                noise_boundary32_list.append(noise_aux["noise_boundary32"])
                fgm_debug["forensic_adapter_reliability_mean"] = float(noise_reliability.detach().float().mean().cpu())

            fgm_gate_mean = 1.0
            if self.reliability_gate_enabled and not disable_fgm and not bool(ablation.get("disable_reliability_gate", False)):
                gate_inputs = self._fgm_gate_inputs(fgm_aux, noise_reliability, b, k, feat_h, feat_w, x_m.device, x_m.dtype)
                gate, gate_debug = self.reliability_gate(gate_inputs)
                f_ip = f_ip * gate.reshape(b, k, 1, feat_h, feat_w)
                gate_regularization_list.append(gate.reshape(b, k, 1, feat_h, feat_w))
                fgm_gate_mean = gate_debug["reliability_gate_mean"]
                fgm_debug.update(gate_debug)
            elif self.reliability_gate_enabled:
                fgm_debug["reliability_gate_mean"] = 0.0

            if self.use_temporal and not (disable_ccm and disable_fgm and not tcu_active):
                f32, fusion_debug = self.fusion(x_m, f_cc, f_ip)
            else:
                f32, fusion_debug = self.static_adapter(x_m)
            if self.forensic_enabled:
                forensic_clip = forensic_feat[:, clip_idx].reshape(b * k, forensic_c, forensic_h, forensic_w)
                if forensic_clip.shape[-2:] != f32.shape[-2:]:
                    forensic_clip = F.interpolate(forensic_clip, size=f32.shape[-2:], mode="bilinear", align_corners=False)
                f32 = f32 + forensic_clip
                fusion_debug["fusion_forensic_norm"] = float(forensic_clip.detach().float().norm().cpu())
            if self.noise_adapter_enabled and not bool(ablation.get("disable_noise_adapter", False)):
                if noise_feat is None:
                    noise_frames = video[:, clip_idx]
                    noise_feat, noise_reliability, noise_aux = self.noise_adapter.extract_features(noise_frames, f32.dtype)
                noise_delta, noise_debug = self.noise_adapter.apply_to_feature(noise_feat, f32)
                f32 = f32 + noise_delta
                noise_debug["forensic_adapter_reliability_mean"] = float(noise_reliability.detach().float().mean().cpu()) if noise_reliability is not None else 0.0
                fusion_debug.update(noise_debug)
                fusion_debug["fusion_noise_delta_norm"] = float(noise_delta.detach().float().norm().cpu())
            if self.prototype_memory_enabled and not bool(ablation.get("disable_prototype_memory", False)):
                proto_delta, proto_debug = self.prototype_memory.read(f32, bank)
                f32 = f32 + proto_delta
                fusion_debug.update(proto_debug)

            if self.sumi_enabled and not bool(ablation.get("disable_sumi", False)):
                adv_cfg = (self.sumi_cfg.get("source_adversarial", {}) or {})
                sumi_aux = self.sumi_heads(f32, grl_lambda=float(adv_cfg.get("grl_lambda", 0.05)))
                sumi_ib_kl_list.append(sumi_aux["sumi_ib_kl_tensor"])
                sumi_source_logits_list.append(sumi_aux["sumi_source_logits"])
                sumi_activation32_list.append(sumi_aux["sumi_activation32"].reshape(b, k, 1, feat_h, feat_w))
                fusion_debug["sumi_ib_kl"] = float(sumi_aux["sumi_ib_kl_tensor"].detach().float().cpu())

            dec = self.decoder(f32)
            logits_clip = dec["logits"].reshape(b, k, 1, h, w)
            mask128_clip = dec["mask128"].reshape(b, k, 1, 128, 128)
            mask256_clip = dec["mask256"].reshape(b, k, 1, 256, 256)
            boundary128 = dec["boundary128"]
            if boundary128 is not None:
                boundary128 = boundary128.reshape(b, k, 1, 128, 128)
            aux_zero_dep = mask128_clip.sum() * 0.0
            if boundary128 is not None:
                aux_zero_dep = aux_zero_dep + boundary128.sum() * 0.0
            if ccm_aux is not None:
                aux_zero_dep = aux_zero_dep + ccm_aux.sum() * 0.0
            if fgm_aux is not None:
                aux_zero_dep = aux_zero_dep + fgm_aux.sum() * 0.0
            logits_clip = logits_clip + aux_zero_dep
            if self.logit_clamp > 0:
                logits_clip = logits_clip.clamp(-self.logit_clamp, self.logit_clamp)

            logits_list.append(logits_clip)
            mask128_list.append(mask128_clip)
            mask256_list.append(mask256_clip)
            boundary128_list.append(boundary128)
            if ccm_aux is not None:
                ccm_aux_list.append(ccm_aux)
            if fgm_aux is not None:
                fgm_aux_list.append(fgm_aux)
            cue_list.append(cue)
            if self.prototype_memory_enabled and not bool(ablation.get("disable_prototype_memory", False)):
                prob32 = torch.sigmoid(mask256_clip.detach().reshape(b * k, 1, 256, 256))
                proto_write_debug = self.prototype_memory.write(
                    f32.detach(),
                    prob32,
                    fgm_gate_mean,
                    bank,
                    epoch=epoch if self.training else None,
                )
                fgm_debug.update(proto_write_debug)
            if not disable_fgm:
                gated_cue, keep_cue, bank_debug = self._quality_gate_cue(cue, logits_clip, bank)
                fgm_debug.update(bank_debug)
                if keep_cue and not (self.prototype_memory_enabled and self.prototype_replace_raw_bank):
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
            "adapter_mask32": torch.stack(adapter_mask32_list, dim=1) if adapter_mask32_list else None,
            "adapter_boundary32": torch.stack(adapter_boundary32_list, dim=1) if adapter_boundary32_list else None,
            "adapter_gate": torch.stack(adapter_gate_list, dim=1) if adapter_gate_list else None,
            "tcu_momentary_mask32": torch.stack(tcu_momentary_list, dim=1) if tcu_momentary_list else None,
            "tcu_gradual_mask32": torch.stack(tcu_gradual_list, dim=1) if tcu_gradual_list else None,
            "tcu_cumulative_mask32": torch.stack(tcu_cumulative_list, dim=1) if tcu_cumulative_list else None,
            "tcu_logit_delta32": torch.stack(tcu_logit_delta_list, dim=1) if tcu_logit_delta_list else None,
            "noise_mask32": torch.stack(noise_mask32_list, dim=1) if noise_mask32_list else None,
            "noise_boundary": torch.stack(noise_boundary32_list, dim=1) if noise_boundary32_list else None,
            "gate_regularization": torch.stack(gate_regularization_list, dim=1) if gate_regularization_list else None,
            "fgm_cue": torch.stack(cue_list, dim=1),
            "sumi_ib_kl_tensor": torch.stack(sumi_ib_kl_list).mean() if sumi_ib_kl_list else None,
            "sumi_source_logits": torch.cat(sumi_source_logits_list, dim=0) if sumi_source_logits_list else None,
            "sumi_activation32": torch.stack(sumi_activation32_list, dim=1) if sumi_activation32_list else None,
            "ttf_residual_energy": temporal_aux.get("ttf_residual_energy"),
            "ttf_patch_hw": temporal_aux.get("ttf_patch_hw"),
            "debug": debug,
        }
        debug.update(self._mean_debug(debug, "reliability_gate_"))
        debug.update(self._mean_debug(debug, "forensic_adapter_"))
        debug.update(self._mean_debug(debug, "prototype_memory_"))
        debug.update(self._mean_debug(debug, "tcu_"))
        debug.update(self._mean_debug(debug, "sumi_"))
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
