from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .lora import (
    add_peft_lora_to_backbone,
    mark_only_lora_trainable,
    parse_lora_layers,
    parse_lora_targets,
)


def load_dinov3_backbone(cfg: dict[str, Any]) -> nn.Module:
    repo = Path(str(cfg.get("repo", "./dinov3"))).expanduser().resolve()
    weights = Path(str(cfg.get("weights", ""))).expanduser().resolve()
    model_name = str(cfg.get("model_name", "dinov3_vitl16"))

    if repo.exists() and (repo / "hubconf.py").exists():
        print(f"[dinov3] loading {model_name} from local repo: {repo}")
        return torch.hub.load(str(repo), model_name, source="local", weights=str(weights))

    if bool(cfg.get("allow_hub_download", False)):
        print(f"[dinov3] local repo not found, fallback to torch.hub github source: {model_name}")
        return torch.hub.load("facebookresearch/dinov3", model_name, weights=str(weights))

    raise FileNotFoundError(
        f"DINOv3 repo not found at {repo}. This new project is independent, "
        "so copy the local dinov3 repo/weights into ./dinov3 or enable allow_hub_download."
    )


class DINOv3B23Encoder(nn.Module):
    """DINOv3 ViT-L/16 final block B23 native patch feature extractor."""

    def __init__(self, dinov3_cfg: dict[str, Any], lora_cfg: dict[str, Any] | None = None):
        super().__init__()
        lora_cfg = lora_cfg or {}
        self.input_size = int(dinov3_cfg.get("input_size", 512))
        self.patch_size = int(dinov3_cfg.get("patch_size", 16))
        self.output_block = int(dinov3_cfg.get("output_block", 23))
        self.output_resolution = int(dinov3_cfg.get("output_resolution", self.input_size // self.patch_size))
        self.freeze_backbone = bool(dinov3_cfg.get("freeze_backbone", True))
        self.use_lora = bool(lora_cfg.get("enabled", False))
        qcfg = dinov3_cfg.get("query_injection", {}) or {}
        self.query_injection_enabled = bool(qcfg.get("enabled", False))
        self.inject_after_block = int(qcfg.get("inject_after_block", 19))
        self.query_blocks = [int(idx) for idx in qcfg.get("query_blocks", [20, 21, 22, 23])]
        self.keep_cls_token = bool(qcfg.get("keep_cls_token", True))
        self.use_block_checkpoint = bool(dinov3_cfg.get("use_activation_checkpoint", False))

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

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return [N, 1024, 32, 32] for [N, 3, 512, 512] RGB input in [0,1]."""
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N,3,H,W], got {tuple(frames.shape)}")
        frames = (frames - self.image_mean) / self.image_std
        grad_enabled = torch.is_grad_enabled() and (not self.freeze_backbone or self.use_lora)
        with torch.set_grad_enabled(grad_enabled):
            out = self.backbone.get_intermediate_layers(
                frames,
                n=[self.output_block],
                reshape=True,
                norm=True,
            )[0]
        if out.shape[-2:] != (self.output_resolution, self.output_resolution):
            raise RuntimeError(
                f"Expected B23 feature resolution {self.output_resolution}x{self.output_resolution}, "
                f"got {tuple(out.shape[-2:])}"
            )
        return out

    def _norm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        storage_count = int(getattr(self.backbone, "n_storage_tokens", 0))
        if bool(getattr(self.backbone, "untie_cls_and_patch_norms", False)):
            cls_reg = self.backbone.cls_norm(tokens[:, : storage_count + 1])
            patch_and_query = self.backbone.norm(tokens[:, storage_count + 1 :])
            return torch.cat((cls_reg, patch_and_query), dim=1)
        return self.backbone.norm(tokens)

    def _run_block(self, block: nn.Module, tokens: torch.Tensor, rope_sincos):
        if self.use_block_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(lambda y, blk=block, rope=rope_sincos: blk(y, rope), tokens, use_reentrant=False)
        return block(tokens, rope_sincos)

    def _extract_patch_tokens(self, native_tokens: torch.Tensor) -> torch.Tensor:
        num_patches = self.output_resolution * self.output_resolution
        if native_tokens.shape[1] < num_patches:
            raise RuntimeError(
                f"Native DINOv3 tokens have only {native_tokens.shape[1]} tokens; "
                f"cannot extract {num_patches} patch tokens."
            )
        patch_tokens = native_tokens[:, -num_patches:, :]
        if patch_tokens.shape[1] != num_patches:
            raise RuntimeError(f"Expected {num_patches} patch tokens, got {patch_tokens.shape[1]}")
        return patch_tokens

    def forward_with_queries(
        self,
        frames: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DINOv3 while appending query tokens through the final transformer blocks."""
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N,3,H,W], got {tuple(frames.shape)}")
        if query_tokens.ndim != 3:
            raise ValueError(f"query_tokens must be [N,Q,C], got {tuple(query_tokens.shape)}")
        if query_tokens.shape[0] != frames.shape[0]:
            raise ValueError(f"query batch {query_tokens.shape[0]} does not match frames batch {frames.shape[0]}")

        frames = (frames - self.image_mean) / self.image_std
        grad_enabled = torch.is_grad_enabled() and (
            not self.freeze_backbone or self.use_lora or query_tokens.requires_grad
        )
        with torch.set_grad_enabled(grad_enabled):
            tokens, patch_hw = self.backbone.prepare_tokens_with_masks(frames)
            h_patches, w_patches = int(patch_hw[0]), int(patch_hw[1])
            if (h_patches, w_patches) != (self.output_resolution, self.output_resolution):
                raise RuntimeError(
                    f"Expected patch grid {self.output_resolution}x{self.output_resolution}, "
                    f"got {h_patches}x{w_patches}"
                )
            if query_tokens.shape[-1] != tokens.shape[-1]:
                raise ValueError(f"query dim {query_tokens.shape[-1]} does not match DINO dim {tokens.shape[-1]}")

            total_blocks = len(self.backbone.blocks)
            output_block = min(self.output_block, total_blocks - 1)
            inject_after = min(self.inject_after_block, output_block)
            rope_sincos = self.backbone.rope_embed(H=h_patches, W=w_patches) if getattr(self.backbone, "rope_embed", None) is not None else None

            for block_idx in range(0, inject_after + 1):
                tokens = self._run_block(self.backbone.blocks[block_idx], tokens, rope_sincos)

            storage_count = int(getattr(self.backbone, "n_storage_tokens", 0))
            prefix_len = storage_count + 1
            q_start = prefix_len
            q_end = prefix_len + query_tokens.shape[1]
            prefix_tokens = tokens[:, :prefix_len]
            patch_tokens = tokens[:, prefix_len:]
            num_patches = h_patches * w_patches
            if patch_tokens.shape[1] != num_patches:
                raise RuntimeError(f"Expected {num_patches} patch tokens, got {patch_tokens.shape[1]}")
            # Insert queries before patch tokens so DINOv3 RoPE still applies to the final patch-token segment.
            tokens = torch.cat((prefix_tokens, query_tokens.to(dtype=tokens.dtype), patch_tokens), dim=1)
            for block_idx in range(inject_after + 1, output_block + 1):
                tokens = self._run_block(self.backbone.blocks[block_idx], tokens, rope_sincos)

            tokens = self._norm_tokens(tokens)
            out_queries = tokens[:, q_start:q_end]
            patch_tokens = self._extract_patch_tokens(tokens)
            patch_features = patch_tokens.transpose(1, 2).reshape(
                frames.shape[0],
                -1,
                self.output_resolution,
                self.output_resolution,
            ).contiguous()
        return patch_features, out_queries
