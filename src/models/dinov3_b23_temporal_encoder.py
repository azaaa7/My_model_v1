from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dinov3_b23_encoder import load_dinov3_backbone
from .lora import (
    add_peft_lora_to_backbone,
    mark_only_lora_trainable,
    parse_lora_layers,
    parse_lora_targets,
)
from .tfcu import TemporalPatchTokenFusion


class DINOv3B23TemporalEncoder(nn.Module):
    """DINOv3 ViT-L/16 B23 encoder with TTF after patch embedding."""

    def __init__(
        self,
        dinov3_cfg: dict[str, Any],
        lora_cfg: dict[str, Any] | None = None,
        temporal_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        lora_cfg = lora_cfg or {}
        temporal_cfg = temporal_cfg or {}
        token_cfg = dict(temporal_cfg.get("token_fusion", {}) or {})
        token_cfg.setdefault("enabled", bool(temporal_cfg.get("enabled", True)))

        self.input_size = int(dinov3_cfg.get("input_size", 512))
        self.patch_size = int(dinov3_cfg.get("patch_size", 16))
        self.output_block = int(dinov3_cfg.get("output_block", 23))
        self.output_resolution = int(dinov3_cfg.get("output_resolution", self.input_size // self.patch_size))
        self.feature_dim = int(dinov3_cfg.get("feature_dim", token_cfg.get("dim", 1024)))
        self.freeze_backbone = bool(dinov3_cfg.get("freeze_backbone", True))
        self.use_lora = bool(lora_cfg.get("enabled", False))
        self.temporal_enabled = bool(temporal_cfg.get("enabled", True)) and bool(token_cfg.get("enabled", True))
        self.use_block_checkpoint = bool(
            temporal_cfg.get("use_activation_checkpoint", temporal_cfg.get("block_checkpoint", False))
        )

        self.backbone = load_dinov3_backbone(dinov3_cfg)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lora_layers = 0
        if self.use_lora:
            targets = parse_lora_targets(lora_cfg.get("targets"))
            layers = parse_lora_layers(lora_cfg.get("layers", "all"))
            self.backbone, self.lora_layers = add_peft_lora_to_backbone(
                self.backbone,
                target_suffixes=targets,
                rank=int(lora_cfg.get("rank", 32)),
                alpha=float(lora_cfg.get("alpha", 64)),
                dropout=float(lora_cfg.get("dropout", 0.1)),
                lora_layers=layers,
            )
            mark_only_lora_trainable(self.backbone)

        token_cfg.setdefault("dim", self.feature_dim)
        self.temporal_fusion = TemporalPatchTokenFusion(token_cfg, dim=self.feature_dim)
        if not self.temporal_enabled:
            for param in self.temporal_fusion.parameters():
                param.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def _norm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        storage_count = int(getattr(self.backbone, "n_storage_tokens", 0))
        if bool(getattr(self.backbone, "untie_cls_and_patch_norms", False)):
            cls_reg = self.backbone.cls_norm(tokens[:, : storage_count + 1])
            patch = self.backbone.norm(tokens[:, storage_count + 1 :])
            return torch.cat((cls_reg, patch), dim=1)
        return self.backbone.norm(tokens)

    def _run_blocks(self, tokens: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        h_patches, w_patches = patch_hw
        for idx, block in enumerate(self.backbone.blocks):
            if getattr(self.backbone, "rope_embed", None) is not None:
                rope_sincos = self.backbone.rope_embed(H=h_patches, W=w_patches)
            else:
                rope_sincos = None
            if self.use_block_checkpoint and self.training and torch.is_grad_enabled():
                tokens = checkpoint(lambda y, blk=block, rope=rope_sincos: blk(y, rope), tokens, use_reentrant=False)
            else:
                tokens = block(tokens, rope_sincos)
            if idx >= self.output_block:
                break
        return tokens

    def forward(self, video: torch.Tensor, return_aux: bool = False):
        """Return B23 features for [B,M,K,3,H,W].

        Output feature shape is [B*M*K, C, 32, 32] for 512x512 inputs.
        """
        if video.ndim != 6:
            raise ValueError(f"video must be [B,M,K,3,H,W], got {tuple(video.shape)}")
        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)
        frames = (frames - self.image_mean) / self.image_std

        grad_enabled = torch.is_grad_enabled() and (
            not self.freeze_backbone or self.use_lora or self.temporal_enabled
        )
        with torch.set_grad_enabled(grad_enabled):
            tokens, patch_hw = self.backbone.prepare_tokens_with_masks(frames)
            h_patches, w_patches = int(patch_hw[0]), int(patch_hw[1])
            num_patches = h_patches * w_patches
            storage_count = int(getattr(self.backbone, "n_storage_tokens", 0))
            extra_tokens = tokens[:, : storage_count + 1]
            patch_tokens = tokens[:, storage_count + 1 :]
            if patch_tokens.shape[1] != num_patches:
                raise RuntimeError(
                    f"Expected {num_patches} patch tokens from DINOv3, got {patch_tokens.shape[1]}"
                )

            patch_tokens_video = patch_tokens.reshape(b, m, k, num_patches, self.feature_dim)
            patch_tokens_video, ttf_debug, ttf_aux = self.temporal_fusion(
                patch_tokens_video,
                return_aux=True,
            )
            patch_tokens = patch_tokens_video.reshape(b * m * k, num_patches, self.feature_dim)
            tokens = torch.cat((extra_tokens, patch_tokens), dim=1)
            tokens = self._run_blocks(tokens, (h_patches, w_patches))
            tokens = self._norm_tokens(tokens)
            patch_tokens = tokens[:, storage_count + 1 :]
            features = patch_tokens.reshape(b * m * k, h_patches, w_patches, self.feature_dim)
            features = features.permute(0, 3, 1, 2).contiguous()

        if features.shape[-2:] != (self.output_resolution, self.output_resolution):
            raise RuntimeError(
                f"Expected B23 feature resolution {self.output_resolution}x{self.output_resolution}, "
                f"got {tuple(features.shape[-2:])}"
            )
        debug = {
            "temporal_encoder_enabled": True,
            "temporal_encoder_type": "DINOv3B23TemporalEncoder",
            "patch_hw": (int(features.shape[-2]), int(features.shape[-1])),
            "feat_shape": tuple(features.shape),
            "output_block": int(self.output_block),
        }
        debug.update(ttf_debug)
        aux = {
            "ttf_residual_energy": ttf_aux.get("ttf_residual_energy"),
            "ttf_patch_hw": (h_patches, w_patches),
        }
        return (features, debug, aux) if return_aux else (features, debug)
