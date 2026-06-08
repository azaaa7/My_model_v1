from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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

