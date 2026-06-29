# B25 No-Gate Spatiotemporal Adapter for Video Inpainting Localization

> Project: `azaaa7/My_model_v1`  
> Goal: replace the current CCM-predecoder direction with a lighter, more stable **no-gate Spatiotemporal Adapter** inspired by CVPR 2025 Plug-and-Play deepfake video detection.  
> Recommended new config: `configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml`  
> Main principle: **keep DINOv3-IML paper loss unchanged; add only a small residual 3D-conv adapter before the decoder.**

---

## 0. Why change direction

Your current result shows:

```text
Baseline:
Average IoU       0.7813
same_source IoU   0.8130
cross_source OPN  0.7180

+CCM:
Average IoU       0.7802
same_source IoU   0.8167
cross_source OPN  0.7072
```

Interpretation:

```text
CCM improves DVI/CPNET slightly,
but hurts OPN cross-source generalization.
```

This suggests the current CCM is learning source-specific temporal correlations rather than general video inpainting artifacts.

The current `CCMLite` is an attention-style module:

```text
DINO feature
  -> GroupNorm
  -> 1x1 projection to low dim
  -> cross-frame multi-head attention
  -> concat with original normalized feature
  -> 1x1 fuse back to 1024
  -> x + alpha_cc * residual
```

This is expressive, but for your current data scale it can overfit to DVI/CPNET temporal statistics. Also, the current `alpha_cc` is trainable, which is effectively a learned residual gate.

---

## 1. What to borrow from CVPR 2025 Plug-and-Play

The CVPR 2025 paper **Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning** proposes a lightweight Spatiotemporal Adapter, or StA.

Important ideas to borrow:

1. Start from a strong pretrained image model.
2. Do not design a full video backbone from scratch.
3. Add a lightweight adapter to convert image features into video-aware features.
4. Use separate spatial and temporal 3D convolution branches:
   - spatial branch: kernel `(1, N, N)`
   - temporal branch: kernel `(N, 1, 1)`
5. Train only the lightweight added modules while keeping the strong pretrained model mostly frozen.

Important ideas **not** to copy directly for this project:

1. Do not use face-specific Facial Feature Drift.
2. Do not use facial organ blending.
3. Do not add cross-attention between branches in the first version.
4. Do not use learned gates.

For your task, the adapted artifact concept is:

```text
Facial Feature Drift  ->  Inpainted Region Feature Drift
```

That is, video inpainting can introduce subtle temporal drift in object boundaries, textures, mask edges, and reconstructed regions.

---

## 2. Proposed architecture

Replace:

```text
DINOv3 feature -> CCM -> DINOv3-IML head
```

with:

```text
DINOv3 feature -> NoGateStA -> DINOv3-IML head
```

Full B25 pipeline:

```text
video: [B, M, K, 3, 512, 512]
  -> flatten frames
  -> DINOv3B23Encoder
  -> features: [B, M, K, 1024, 32, 32]
  -> for each clip:
       NoGateSpatiotemporalAdapter
         down: Conv3d 1x1x1, 1024 -> 128
         spatial branch: depthwise Conv3d (1,3,3)
         temporal branch: depthwise Conv3d (3,1,1)
         optional multiscale: 3/5/7
         fixed average fusion: 0.5 * (spatial + temporal)
         up: Conv3d 1x1x1, 128 -> 1024
         residual: x + up(...)
  -> DINOv3IMLHead
  -> logits: [B, M, K, 1, 512, 512]
  -> original paper loss
```

There is no learned gate.

The only trainable components are:

```text
LoRA parameters
NoGateStA adapter parameters
DINOv3IMLHead parameters
```

Loss remains:

```text
loss = BCEWithLogits(logits, mask)
     + 20 * BCEWithLogits(logits, mask, weight=edge_mask)
```

---

## 3. Files to add or modify

```text
src/models/adapters/nogate_spatiotemporal_adapter.py
src/models/adapters/__init__.py
src/models/b25_dinov3_iml_nogate_sta_video_model.py
src/models/__init__.py
src/models/builder.py
configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml
tools/smoke_b25_nogate_sta.py
```

This document assumes you already added these from the previous B24 implementation:

```text
src/models/decoders/dinov3_iml_head.py
src/losses/dinov3_iml_original_loss.py
```

---

# Part A. No-gate adapter implementation

## A1. Add `src/models/adapters/nogate_spatiotemporal_adapter.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _make_norm3d(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm in ["none", "identity"]:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


class DepthwiseConv3dBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int, int],
        norm: str = "gn",
        act: str = "gelu",
    ):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)

        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported act: {act}")

        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            _make_norm3d(norm, channels),
            activation,
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NoGateSpatiotemporalAdapter(nn.Module):
    """
    No-gate StA-lite adapter.

    Input:
        x: [B, K, C, H, W]

    No learned gate.
    No cross-attention.
    No sigmoid gating.
    No alpha parameter.

    Identity-safe design:
        final up projection can be zero-initialized, so the adapter starts as exact identity.
    """

    def __init__(self, in_channels: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", True))
        self.in_channels = int(in_channels)
        self.bottleneck = int(cfg.get("bottleneck", 128))
        self.temporal_kernel = int(cfg.get("temporal_kernel", 3))
        self.spatial_kernel = int(cfg.get("spatial_kernel", 3))
        self.multiscale = bool(cfg.get("multiscale", False))
        self.norm = str(cfg.get("norm", "gn"))
        self.act = str(cfg.get("act", "gelu"))
        self.zero_init_up = bool(cfg.get("zero_init_up", True))
        self.drop_path_prob = float(cfg.get("drop_path_prob", 0.0))

        self.down = nn.Sequential(
            nn.Conv3d(self.in_channels, self.bottleneck, kernel_size=1, bias=False),
            _make_norm3d(self.norm, self.bottleneck),
            nn.GELU() if self.act == "gelu" else nn.ReLU(inplace=True),
        )

        if self.multiscale:
            spatial_kernels = [3, 5, 7]
            temporal_kernels = [3, 5, 7]
        else:
            spatial_kernels = [self.spatial_kernel]
            temporal_kernels = [self.temporal_kernel]

        self.spatial_blocks = nn.ModuleList(
            [
                DepthwiseConv3dBlock(
                    self.bottleneck,
                    kernel_size=(1, k, k),
                    norm=self.norm,
                    act=self.act,
                )
                for k in spatial_kernels
            ]
        )

        self.temporal_blocks = nn.ModuleList(
            [
                DepthwiseConv3dBlock(
                    self.bottleneck,
                    kernel_size=(k, 1, 1),
                    norm=self.norm,
                    act=self.act,
                )
                for k in temporal_kernels
            ]
        )

        self.up = nn.Conv3d(self.bottleneck, self.in_channels, kernel_size=1, bias=True)

        if self.zero_init_up:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def _drop_path(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob <= 0.0:
            return residual
        keep_prob = 1.0 - self.drop_path_prob
        shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        random_tensor = residual.new_empty(shape).bernoulli_(keep_prob)
        return residual / keep_prob * random_tensor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return x, {"nogate_sta_enabled": False}

        if x.ndim != 5:
            raise ValueError(f"NoGateSpatiotemporalAdapter expects [B,K,C,H,W], got {tuple(x.shape)}")

        b, k, c, h, w = x.shape

        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.down(x_3d)

        spatial = 0.0
        for block in self.spatial_blocks:
            spatial = spatial + block(z)
        spatial = spatial / len(self.spatial_blocks)

        temporal = 0.0
        for block in self.temporal_blocks:
            temporal = temporal + block(z)
        temporal = temporal / len(self.temporal_blocks)

        # Fixed average, not a learned gate.
        fused = 0.5 * (spatial + temporal)

        residual = self.up(fused)
        residual = self._drop_path(residual)

        out_3d = x_3d + residual
        out = out_3d.permute(0, 2, 1, 3, 4).contiguous()

        debug = {
            "nogate_sta_enabled": True,
            "input_shape": tuple(x.shape),
            "bottleneck": self.bottleneck,
            "multiscale": self.multiscale,
            "zero_init_up": self.zero_init_up,
            "drop_path_prob": self.drop_path_prob,
            "residual_norm": float(residual.detach().float().norm().cpu()),
            "input_norm": float(x_3d.detach().float().norm().cpu()),
        }
        return out, debug
```

---

## A2. Add `src/models/adapters/__init__.py`

```python
from .nogate_spatiotemporal_adapter import NoGateSpatiotemporalAdapter

__all__ = [
    "NoGateSpatiotemporalAdapter",
]
```

---

# Part B. B25 model implementation

## B1. Add `src/models/b25_dinov3_iml_nogate_sta_video_model.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .adapters import NoGateSpatiotemporalAdapter
from .decoders import DINOv3IMLHead
from .dinov3_b23_encoder import DINOv3B23Encoder


class B25DINOv3IMLNoGateStAVideoModel(nn.Module):
    """
    B25:
        DINOv3-B23 + LoRA QKV + NoGate StA-lite + DINOv3-IML head.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        dinov3_cfg = cfg.get("dinov3", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        adapter_cfg = cfg.get("nogate_sta", {}) or {}
        decoder_cfg = dict(cfg.get("decoder", {}) or {})

        self.feature_dim = int(dinov3_cfg.get("feature_dim", 1024))
        self.input_size = int(cfg.get("input_size", dinov3_cfg.get("input_size", 512)))
        self.encoder_chunk = int(cfg.get("encoder_chunk", 0) or 0)
        self.use_activation_checkpoint = bool(cfg.get("use_activation_checkpoint", False))

        self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)

        adapter_cfg.setdefault("enabled", True)
        self.nogate_sta = NoGateSpatiotemporalAdapter(
            in_channels=self.feature_dim,
            cfg=adapter_cfg,
        )

        decoder_cfg.setdefault("type", "dinov3_iml_head")
        decoder_cfg.setdefault("in_channels", self.feature_dim)
        decoder_cfg.setdefault("hidden1", self.feature_dim // 2)
        decoder_cfg.setdefault("hidden2", self.feature_dim // 4)
        decoder_cfg.setdefault("image_size", self.input_size)
        decoder_cfg.setdefault("norm", "bn")
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
        if video.ndim == 5:
            video = video[:, None]

        if video.ndim != 6:
            raise ValueError(
                f"video must be [B,M,K,3,H,W] or [B,K,3,H,W], got {tuple(video.shape)}"
            )

        ablation = ablation or {}
        disable_sta = bool(ablation.get("disable_sta", False))

        b, m, k, c, h, w = video.shape
        frames = video.reshape(b * m * k, c, h, w)

        feat = self.encode_frames(frames)
        _, feat_c, feat_h, feat_w = feat.shape
        features = feat.reshape(b, m, k, feat_c, feat_h, feat_w)

        logits_list = []
        logits32_list = []

        debug: dict[str, Any] = {
            "model": "B25DINOv3IMLNoGateStAVideoModel",
            "input_video_shape": tuple(video.shape),
            "feature_shape": tuple(features.shape),
            "disable_sta": disable_sta,
        }

        for clip_idx in range(m):
            x_clip = features[:, clip_idx]  # [B,K,C,32,32]

            if disable_sta:
                x_adapt = x_clip
                sta_debug = {"nogate_sta_enabled": False, "disabled_by_ablation": True}
            else:
                x_adapt, sta_debug = self.nogate_sta(x_clip)

            x_flat = x_adapt.reshape(b * k, feat_c, feat_h, feat_w)
            dec = self.decoder(x_flat, output_size=(h, w))

            logits_clip = dec["logits"].reshape(b, k, 1, h, w)
            logits32_clip = dec["logits32"].reshape(b, k, 1, feat_h, feat_w)

            logits_list.append(logits_clip)
            logits32_list.append(logits32_clip)

            debug[f"clip{clip_idx}_nogate_sta"] = sta_debug
            debug[f"clip{clip_idx}_decoder"] = dec["debug"]

        logits = torch.stack(logits_list, dim=1)
        logits32 = torch.stack(logits32_list, dim=1)

        return {
            "logits": logits,
            "aux": {
                "logits32": logits32,
                "debug": debug,
            },
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

---

## B2. Register model

### `src/models/__init__.py`

```python
from .b25_dinov3_iml_nogate_sta_video_model import B25DINOv3IMLNoGateStAVideoModel
```

### `src/models/builder.py`

```python
if name == "B25DINOv3IMLNoGateStAVideoModel":
    return B25DINOv3IMLNoGateStAVideoModel(cfg)
```

---

# Part C. Loss

Do not add a new loss.

Use the same original DINOv3-IML loss from B24:

```yaml
loss:
  name: "dinov3_iml_original"
  edge_lambda: 20.0
  edge_mask_width: 7
```

No Dice.

No Tversky.

No CCM loss.

No adapter loss.

No auxiliary temporal loss.

The adapter is trained only through final segmentation logits.

---

# Part D. Recommended config

## Add `configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml`

```yaml
type: train
seed: 666666

input_size: 512
batch_size: 1

num_clips: 4
num_frames: 4
clip_stride: 1
encoder_chunk: 2

gt_ratio: 1

train_samples:
  - "./flist/DAVIS-VI_tra_DVI_30.npy"
  - "./flist/DAVIS-VI_tra_CPNET_30.npy"

val_samples:
  - "./flist/DAVIS-VI_val_DVI_20.npy"
  - "./flist/DAVIS-VI_val_CPNET_20.npy"

test_samples:
  - "./flist/DAVIS-VI_val_DVI_20.npy"
  - "./flist/DAVIS-VI_val_CPNET_20.npy"
  - "./flist/DAVIS-VI_val_OPN_20.npy"

augment_prob: 0.75
spatial_augment_prob: 0.75
appearance_augment_prob: 0.50

amp: true
use_activation_checkpoint: true
grad_accum_steps: 8
num_workers: 4
log_interval: 20

model:
  name: "B25DINOv3IMLNoGateStAVideoModel"

dinov3:
  repo: "./dinov3"
  weights: "./dinov3/dinov3_weight/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  model_name: "dinov3_vitl16"
  input_size: 512
  patch_size: 16
  output_block: 23
  output_resolution: 32
  feature_dim: 1024
  freeze_backbone: true
  allow_hub_download: false

lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.0
  layers: "all"
  targets: "attn.qkv"

nogate_sta:
  enabled: true
  bottleneck: 128
  multiscale: false
  spatial_kernel: 3
  temporal_kernel: 3
  norm: "gn"
  act: "gelu"
  zero_init_up: true
  drop_path_prob: 0.05

decoder:
  type: "dinov3_iml_head"
  in_channels: 1024
  hidden1: 512
  hidden2: 256
  norm: "bn"
  image_size: 512

loss:
  name: "dinov3_iml_original"
  edge_lambda: 20.0
  edge_mask_width: 7

optimizer:
  learning_rate: 3.0e-4
  lr_lora: 3.0e-4
  lr_adapter: 1.0e-4
  lr_decoder: 3.0e-4
  weight_decay: 5.0e-2

scheduler:
  type: "cosine"
  warmup_epochs: 5
  min_lr: 1.0e-6

train:
  n_epochs: 100
  save_dir: "runs/b25_dinov3_iml_nogate_sta_paperloss_lora32"
  val_interval: 5

validation:
  val_full_video: true
  max_batches: null
  val_test_max_clips: null

ddp:
  auto_torchrun: true
  cuda_visible_devices: "4,5"
  nproc_per_node: 2
  dist_backend: "nccl"
  find_unused_parameters: false
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  torchrun_log_dir: "runs/b25_dinov3_iml_nogate_sta_paperloss_lora32/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
```

---

# Part E. Optimizer parameter groups

Your optimizer builder should recognize adapter parameters.

Add logic similar to:

```python
if "lora_" in name or ".lora_" in name:
    group_name = "lora"
elif name.startswith("nogate_sta.") or ".nogate_sta." in name:
    group_name = "adapter"
elif name.startswith("decoder.") or ".decoder." in name:
    group_name = "decoder"
else:
    group_name = "other"
```

Recommended learning rates:

```text
LoRA      3e-4
Adapter   1e-4
Decoder   3e-4
Other     3e-4 or existing default
```

Adapter is new and directly modifies the DINO feature before the head, so do not start too high.

---

# Part F. Smoke test

## Add `tools/smoke_b25_nogate_sta.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.models.builder import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg).to(args.device).train()

    b = 1
    m = 1
    k = 4
    h = int(cfg.get("input_size", 512))
    w = h

    video = torch.rand(b, m, k, 3, h, w, device=args.device)

    with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
        out = model(video)

    print("logits:", tuple(out["logits"].shape))
    print("logits32:", tuple(out["aux"]["logits32"].shape))

    assert out["logits"].shape == (b, m, k, 1, h, w)
    assert out["aux"]["logits32"].shape == (b, m, k, 1, 32, 32)
    assert torch.isfinite(out["logits"]).all()

    groups = {"lora": 0, "adapter": 0, "decoder": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if "lora_" in name or ".lora_" in name:
            groups["lora"] += n
        elif "nogate_sta" in name:
            groups["adapter"] += n
        elif "decoder" in name:
            groups["decoder"] += n
        else:
            groups["other"] += n

    for key, val in groups.items():
        print(f"trainable[{key}] = {val:,}")

    debug = out["aux"]["debug"]
    if "clip0_nogate_sta" in debug:
        print("adapter debug:", debug["clip0_nogate_sta"])

    print("B25 NoGate StA smoke test passed.")


if __name__ == "__main__":
    main()
```

Run:

```bash
python tools/smoke_b25_nogate_sta.py   --config configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml
```

---

# Part G. Training ablations

Run these in order.

## G1. B24 baseline

```text
DINOv3 + LoRA + DINOv3IMLHead + original paper loss
No CCM
No adapter
```

## G2. B24 + CCM

Already tested. Keep it as reference.

## G3. B25 NoGate StA single-scale

```yaml
nogate_sta:
  enabled: true
  bottleneck: 128
  multiscale: false
  spatial_kernel: 3
  temporal_kernel: 3
  zero_init_up: true
  drop_path_prob: 0.05
```

This is the recommended first B25.

## G4. B25 NoGate StA multiscale

```yaml
nogate_sta:
  multiscale: true
```

Only try it after single-scale is stable.

## G5. Decoder norm ablation

Strict paper:

```yaml
decoder:
  norm: "bn"
```

Video stable:

```yaml
decoder:
  norm: "gn"
```

---

# Part H. Optional data augmentation inspired by Video-Level Blending

The CVPR 2025 paper uses Video-Level Blending to simulate facial feature drift. Your task is not face deepfake, so do not copy facial-region blending.

A compatible later-stage adaptation is **Region Temporal Drift Augmentation**:

```text
For frames inside a clip:
  choose manipulated region mask
  apply small random affine / elastic shift to the region feature or RGB patch
  blend it back only inside mask or near mask boundary
```

Recommended later, not first version:

```yaml
temporal_drift_aug:
  enabled: false
  prob: 0.20
  max_translate: 3
  max_rotate: 2
  boundary_only: true
```

Do not implement this before B25 single-scale adapter is tested.

---

# Part I. Evaluation protocol

Always report:

```text
DVI_20
CPNET_20
OPN_20
Average
same_source_avg
cross_source_opn
```

Also add threshold sweep:

```text
threshold = 0.30, 0.35, 0.40, 0.45, 0.50, 0.55
```

Reason:

Your CCM result showed OPN precision increased but recall dropped. That can mean calibration drift. B25 should be evaluated both at fixed threshold and best validation threshold.

Add metrics if missing:

```text
AUC
AP
best-F1 threshold
best-IoU threshold
```

---

# Part J. Expected outcomes

## If B25 improves OPN

Keep this direction.

Reason:

```text
Local 3D-conv adapter learns general temporal drift without source-specific attention matching.
```

## If B25 improves same-source but hurts OPN

Reduce adapter capacity:

```yaml
nogate_sta:
  bottleneck: 64
  multiscale: false
  drop_path_prob: 0.10

optimizer:
  lr_adapter: 5.0e-5
```

## If B25 does nothing

Try multiscale:

```yaml
nogate_sta:
  multiscale: true
```

Then try temporal-only as a diagnostic.

## If B25 is worse everywhere

Disable zero-init to check learning speed:

```yaml
nogate_sta:
  zero_init_up: false
```

or train longer with warmup.

---

# Part K. One-shot agent instruction

```text
Modify My_model_v1 to add a no-gate spatiotemporal adapter model inspired by CVPR 2025 Plug-and-Play StA, but adapted for video inpainting localization.

Do not use CCM.
Do not use cross-attention.
Do not use learned gates.
Do not add auxiliary losses.

Required changes:

1. Add src/models/adapters/nogate_spatiotemporal_adapter.py
   - Implement NoGateSpatiotemporalAdapter.
   - Input [B,K,C,H,W].
   - Convert to [B,C,K,H,W].
   - 1x1x1 Conv3d down: C -> bottleneck.
   - Spatial depthwise Conv3d branch with kernel (1,3,3).
   - Temporal depthwise Conv3d branch with kernel (3,1,1).
   - Fuse by fixed average: 0.5*(spatial+temporal).
   - 1x1x1 Conv3d up: bottleneck -> C.
   - Residual add: x + residual.
   - No alpha parameter, no sigmoid gate, no learned branch weights.
   - Zero-init the final up projection by default.

2. Add src/models/adapters/__init__.py and export NoGateSpatiotemporalAdapter.

3. Add src/models/b25_dinov3_iml_nogate_sta_video_model.py
   - Use DINOv3B23Encoder.
   - Use DINOv3IMLHead.
   - Insert NoGateSpatiotemporalAdapter before the decoder.
   - Accept [B,M,K,3,H,W] and [B,K,3,H,W].
   - Output logits [B,M,K,1,H,W].
   - Add ablation support: model(video, ablation={"disable_sta": True}).

4. Register B25DINOv3IMLNoGateStAVideoModel in:
   - src/models/__init__.py
   - src/models/builder.py

5. Use existing DINOv3IMLOriginalLoss:
   - loss = BCEWithLogits + 20 * edge-weighted BCEWithLogits
   - no Dice, no Tversky, no adapter loss, no CCM loss.

6. Add configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml.

7. Update optimizer parameter grouping:
   - lora -> lr_lora
   - nogate_sta -> lr_adapter
   - decoder -> lr_decoder

8. Add tools/smoke_b25_nogate_sta.py
   - random input [1,1,4,3,512,512]
   - assert output [1,1,4,1,512,512]
   - print trainable parameter groups
   - print adapter residual_norm / input_norm debug values.
```

---

# Part L. Recommended experiment names

```text
b24_dinov3_iml_video_paperloss_lora32
b24_dinov3_iml_ccm_video_paperloss_lora32
b25_dinov3_iml_nogate_sta_paperloss_lora32
b25_dinov3_iml_nogate_sta_multiscale_paperloss_lora32
b25_dinov3_iml_nogate_sta_gn_paperloss_lora32
```

---

# Part M. Summary

The recommended next direction is:

```text
Use DINOv3-IML as the strong image baseline.
Remove CCM attention from the main path.
Add a small no-gate 3D-conv spatiotemporal adapter before the decoder.
Keep original paper loss unchanged.
Evaluate same-source and cross-source separately.
```

This gives you a cleaner test of whether **local spatiotemporal inconsistency** helps video inpainting localization, without introducing trainable gates, attention overfitting, or extra auxiliary losses.
