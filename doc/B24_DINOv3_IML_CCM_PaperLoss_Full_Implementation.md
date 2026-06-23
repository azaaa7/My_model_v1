# B24 DINOv3-IML Video + CCM Pre-Decoder + Original Paper Loss 实现文档

> 适用项目：`azaaa7/My_model_v1`  
> 目标方向：video inpainting / video manipulation localization  
> 新增模型建议名：`B24DINOv3IMLCCMVideoModel`  
> 推荐配置名：`configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml`

---

## 0. 一句话目标

在你当前项目里新增一个 **B24** 实验分支：

```text
video frames
  -> DINOv3 ViT-L/16 B23 dense feature
  -> frozen backbone + LoRA on QKV
  -> optional CCM-Lite before decoder
  -> DINOv3-IML paper-style 3-conv segmentation head
  -> original paper loss: BCE + 20 * edge-aware BCE
```

重点是：

1. **结构上复现原图论文的核心路线**：DINOv3 frozen backbone + LoRA QKV + 3-conv head。
2. **loss 严格使用原始论文 loss**：`BCEWithLogits + edge_lambda * edge-weighted BCEWithLogits`。
3. **视频上只在 decoder 前加入 CCM**，不再引入额外 aux loss。
4. **保留你项目里效果好的 b23_ccm_lite_lora32 的视频采样 recipe**：`num_clips=4, num_frames=4`，而不是退回短上下文。

---

## 1. 已确认的项目基础

你当前项目已经有这些可复用模块：

```text
src/models/dinov3_b23_encoder.py
src/models/tfcu/ccm_lite.py
src/models/tfcu/fusion.py
src/models/decoders/lite_boundary_decoder.py
src/models/b23_tfcu_ccm_fgm_model.py
src/models/builder.py
```

其中：

- `DINOv3B23Encoder` 已经能输出 `[N, 1024, 32, 32]` 的 DINOv3 ViT-L/16 B23 dense feature。
- `b23_ccm_lite_lora32.yml` 已经验证过一个很强的 recipe：
  - `num_clips: 4`
  - `num_frames: 4`
  - DINOv3 ViT-L/16
  - LoRA rank 32
  - CCM-Lite
  - strong augmentation
- 当前 `LiteBoundaryDecoder` 是 `32 -> 64 -> 128 -> 256 -> 512` 的轻量 decoder，其中 512 输出只是 1-channel logit 上采样。
- 本次 B24 不直接替换旧 B23，而是新增一个干净实验分支，便于对照。

---

## 2. 原始论文结构与本项目适配

原始 DINOv3-IML 论文结构是：

```text
image
  -> DINOv3 ViT backbone
  -> LoRA only on QKV
  -> dense patch feature, e.g. [B, 1024, 32, 32]
  -> 3-conv segmentation head:
       Conv3x3: 1024 -> 512
       BN + ReLU
       Conv3x3: 512 -> 256
       BN + ReLU
       Conv1x1: 256 -> 1
  -> bilinear upsample to 512
```

视频版适配为：

```text
video: [B, M, K, 3, 512, 512]
  -> flatten frames: [B*M*K, 3, 512, 512]
  -> DINOv3B23Encoder
  -> features: [B, M, K, 1024, 32, 32]
  -> for each clip M:
       optional CCM-Lite over K frames
       decode each frame with DINOv3IMLHead
  -> logits: [B, M, K, 1, 512, 512]
```

这里的 CCM 是 **pre-decoder residual feature module**，不是新的监督头。也就是说，CCM 只改变送入 3-conv head 的 feature，不新增 `ccm_mask32` loss。

---

## 3. 实验命名建议

建议保留两个核心实验：

| 实验 | 配置名 | 目的 |
|---|---|---|
| B24-A | `b24_dinov3_iml_video_paperloss_lora32.yml` | 严格把原始 image baseline 扩展到 video frames，不加 CCM |
| B24-B | `b24_dinov3_iml_ccm_video_paperloss_lora32.yml` | 在 decoder 前加入 CCM，但 loss 仍保持原论文 loss |

后续只做 ablation，不要一开始同时改 loss、decoder、采样和 LoRA 范围。

---

## 4. 新增文件总览

需要新增或修改：

```text
src/models/decoders/dinov3_iml_head.py
src/models/b24_dinov3_iml_ccm_video_model.py
src/losses/dinov3_iml_original_loss.py
src/models/decoders/__init__.py
src/models/__init__.py
src/models/builder.py
configs/b24_dinov3_iml_video_paperloss_lora32.yml
configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml
tools/smoke_b24_dinov3_iml_ccm.py
```

---

# Part A. Decoder 实现

## A1. 新增 `src/models/decoders/dinov3_iml_head.py`

这个 head 尽量贴近原始论文：

- 输入：`[N, 1024, 32, 32]`
- 输出：`[N, 1, 512, 512]`
- 训练时返回 logits，不做 sigmoid
- 默认使用 `BatchNorm2d` 以贴近原始论文
- 视频小 batch 不稳时，可配置为 `GroupNorm`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in ["none", "identity"]:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


class DINOv3IMLHead(nn.Module):
    """
    DINOv3-IML paper-style segmentation head.

    Original image version:
        feat_dim -> feat_dim/2 -> feat_dim/4 -> 1

    For ViT-L/16:
        1024 -> 512 -> 256 -> 1

    This module returns logits. Apply sigmoid only for visualization/inference.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}

        in_channels = int(cfg.get("in_channels", 1024))
        hidden1 = int(cfg.get("hidden1", in_channels // 2))
        hidden2 = int(cfg.get("hidden2", in_channels // 4))
        norm = str(cfg.get("norm", "bn"))
        self.image_size = int(cfg.get("image_size", cfg.get("input_size", 512)))

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=True),
            _make_norm(norm, hidden1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=True),
            _make_norm(norm, hidden2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1, bias=True),
        )

        self._init_seg_head()

    def _init_seg_head(self) -> None:
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Args:
            features: [N, C, 32, 32]
            output_size: normally (512, 512)

        Returns:
            dict with logits [N, 1, H, W]
        """
        if features.ndim != 4:
            raise ValueError(f"features must be [N,C,H,W], got {tuple(features.shape)}")

        if output_size is None:
            output_size = (self.image_size, self.image_size)

        logits32 = self.seg_head(features)
        logits = F.interpolate(
            logits32,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": logits,
            "logits32": logits32,
            "debug": {
                "decoder_type": "DINOv3IMLHead",
                "decoder_input_shape": tuple(features.shape),
                "decoder_logits32_shape": tuple(logits32.shape),
                "decoder_logits_shape": tuple(logits.shape),
            },
        }
```

---

## A2. 修改 `src/models/decoders/__init__.py`

追加导出：

```python
from .dinov3_iml_head import DINOv3IMLHead
```

如果当前文件里有 `__all__`，也把它加进去：

```python
__all__ = [
    "LiteBoundaryDecoder",
    "DINOv3IMLHead",
]
```

---

# Part B. B24 视频模型实现

## B1. 新增 `src/models/b24_dinov3_iml_ccm_video_model.py`

这个模型做三件事：

1. 使用项目现有 `DINOv3B23Encoder` 抽取 B23 feature。
2. 可选使用 `CCMLite` 在每个 clip 内做跨帧相关性增强。
3. 用 DINOv3-IML paper head 输出 logits。

注意：

- 不使用 `LiteBoundaryDecoder`。
- 不使用 `mask128/boundary128/ccm_mask32` aux loss。
- CCM 的 aux 输出可以保留在 `debug` 中，但不进入 loss。

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoders import DINOv3IMLHead
from .dinov3_b23_encoder import DINOv3B23Encoder
from .tfcu import CCMLite


class B24DINOv3IMLCCMVideoModel(nn.Module):
    """
    B24 video model:
        DINOv3-B23 + LoRA QKV + optional CCM-Lite + DINOv3-IML 3-conv head.

    Input:
        video: [B, M, K, 3, H, W] or [B, K, 3, H, W]

    Output:
        {
            "logits": [B, M, K, 1, H, W],
            "aux": {
                "logits32": [B, M, K, 1, 32, 32],
                "debug": dict,
                "ccm_mask32": optional/debug only
            }
        }
    """

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

    def _run_ccm(
        self,
        x_clip: torch.Tensor,
        disable_ccm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """
        Args:
            x_clip: [B, K, C, 32, 32]

        Returns:
            feature_for_decoder: [B, K, C, 32, 32]
            ccm_aux: optional [B, K, 1, 32, 32]
            ccm_debug: dict
        """
        if disable_ccm or not self.ccm_enabled:
            return x_clip, None, {
                "ccm_enabled": False,
                "ccm_disabled_by_ablation": bool(disable_ccm),
            }

        out = self.ccm(x_clip)

        # Current project CCMLite returns:
        #   f_cc, ccm_feat, ccm_aux, ccm_debug
        if isinstance(out, tuple) and len(out) >= 4:
            f_cc, ccm_feat, ccm_aux, ccm_debug = out[:4]
        elif isinstance(out, tuple) and len(out) == 3:
            f_cc, ccm_feat, ccm_aux = out
            ccm_debug = {"ccm_enabled": True}
        else:
            raise RuntimeError("Unexpected CCMLite output format.")

        return f_cc, ccm_aux, ccm_debug

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
            x_clip = features[:, clip_idx]  # [B,K,C,32,32]

            x_dec, ccm_aux, ccm_debug = self._run_ccm(
                x_clip,
                disable_ccm=disable_ccm,
            )

            # Decode every frame independently with the paper head.
            x_dec_flat = x_dec.reshape(b * k, feat_c, feat_h, feat_w)
            dec = self.decoder(x_dec_flat, output_size=(h, w))

            logits_clip = dec["logits"].reshape(b, k, 1, h, w)
            logits32_clip = dec["logits32"].reshape(b, k, 1, feat_h, feat_w)

            logits_list.append(logits_clip)
            logits32_list.append(logits32_clip)

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

        return {
            "logits": logits,
            "aux": aux,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

---

## B2. 修改 `src/models/__init__.py`

追加：

```python
from .b24_dinov3_iml_ccm_video_model import B24DINOv3IMLCCMVideoModel
```

---

## B3. 修改 `src/models/builder.py`

在 import 里加入：

```python
from . import B24DINOv3IMLCCMVideoModel
```

在 `build_model` 里加入：

```python
if name == "B24DINOv3IMLCCMVideoModel":
    return B24DINOv3IMLCCMVideoModel(cfg)
```

完整参考：

```python
from __future__ import annotations

from typing import Any

from . import (
    B23TFCUCCMFGMLiteModel,
    B23TemporalRelayLiteModel,
    B23VideoMTWindowModel,
    B24DINOv3IMLCCMVideoModel,
)


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", "B23TFCUCCMFGMLiteModel"))

    if name == "B24DINOv3IMLCCMVideoModel":
        return B24DINOv3IMLCCMVideoModel(cfg)

    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)

    if name == "B23TemporalRelayLiteModel":
        return B23TemporalRelayLiteModel(cfg)

    if name == "B23TFCUCCMFGMLiteModel":
        return B23TFCUCCMFGMLiteModel(cfg)

    raise ValueError(f"Unknown model name: {name}")
```

---

# Part C. 原始论文 loss 实现

## C1. 原始 loss 公式

严格使用：

```text
L = L_BCE + lambda * L_edge
```

其中：

```text
L_BCE  = BCEWithLogits(logits, mask)
L_edge = BCEWithLogits(logits, mask, weight=edge_mask)
lambda = 20.0
edge_mask_width = 7
```

注意：

- 不用 Dice。
- 不用 Tversky。
- 不用 boundary head loss。
- 不用 mask128 aux loss。
- 不用 ccm_mask32 aux loss。
- decoder forward 返回 logits，训练时不要 sigmoid。
- 推理/可视化时才 `torch.sigmoid(logits)`。

---

## C2. 新增 `src/losses/dinov3_iml_original_loss.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common video/image mask or logit formats to [N,1,H,W].

    Supported:
        [B,M,K,1,H,W]
        [B,M,K,H,W]
        [B,K,1,H,W]
        [B,K,H,W]
        [N,1,H,W]
        [N,H,W]
    """
    if x.ndim == 6:
        b, m, k, c, h, w = x.shape
        return x.reshape(b * m * k, c, h, w)

    if x.ndim == 5:
        # [B,K,1,H,W]
        if x.shape[2] == 1:
            b, k, c, h, w = x.shape
            return x.reshape(b * k, c, h, w)
        # [B,M,K,H,W]
        b, m, k, h, w = x.shape
        return x.reshape(b * m * k, 1, h, w)

    if x.ndim == 4:
        # [N,1,H,W]
        if x.shape[1] == 1:
            return x
        # [B,K,H,W]
        b, k, h, w = x.shape
        return x.reshape(b * k, 1, h, w)

    if x.ndim == 3:
        n, h, w = x.shape
        return x.reshape(n, 1, h, w)

    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def make_edge_mask(mask: torch.Tensor, width: int = 7) -> torch.Tensor:
    """
    Build boundary-region weight map from a binary manipulation mask.

    Args:
        mask: [N,1,H,W]
        width: boundary width in pixels. Paper default is 7.

    Returns:
        edge_mask: [N,1,H,W], 1 near boundary, 0 elsewhere.
    """
    if width <= 0:
        return torch.zeros_like(mask)

    mask = (mask > 0.5).float()
    kernel = int(width)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2

    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0.0, 1.0)
    return edge


class DINOv3IMLOriginalLoss(nn.Module):
    """
    Original DINOv3-IML loss:
        BCEWithLogits(logits, mask)
        + edge_lambda * BCEWithLogits(logits, mask, weight=edge_mask)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.edge_lambda = float(cfg.get("edge_lambda", 20.0))
        self.edge_mask_width = int(cfg.get("edge_mask_width", 7))
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]

        if "mask" in batch:
            mask = batch["mask"]
        elif "masks" in batch:
            mask = batch["masks"]
        elif "gt_mask" in batch:
            mask = batch["gt_mask"]
        elif "label_mask" in batch:
            mask = batch["label_mask"]
        else:
            raise KeyError(
                "Cannot find target mask in batch. Expected one of: "
                "mask, masks, gt_mask, label_mask."
            )

        logits = _flatten_to_nchw(logits)
        mask = _flatten_to_nchw(mask).float()

        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")

        predict_loss = self.bce(logits, mask)

        if "edge_mask" in batch and batch["edge_mask"] is not None:
            edge_mask = _flatten_to_nchw(batch["edge_mask"]).float()
            if edge_mask.shape[-2:] != logits.shape[-2:]:
                edge_mask = F.interpolate(edge_mask, size=logits.shape[-2:], mode="nearest")
        else:
            edge_mask = make_edge_mask(mask, width=self.edge_mask_width)

        edge_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=mask,
            weight=edge_mask,
        ) * self.edge_lambda

        total = predict_loss + edge_loss

        return {
            "loss": total,
            "backward_loss": total,
            "predict_loss": predict_loss.detach(),
            "edge_loss": edge_loss.detach(),
            "combined_loss": total.detach(),
        }
```

---

## C3. 注册 loss

如果项目有 `src/losses/__init__.py`，加入：

```python
from .dinov3_iml_original_loss import DINOv3IMLOriginalLoss
```

如果项目有 loss builder，加入：

```python
from src.losses.dinov3_iml_original_loss import DINOv3IMLOriginalLoss


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {}) or {}
    name = str(loss_cfg.get("name", ""))

    if name in ["dinov3_iml_original", "DINOv3IMLOriginalLoss"]:
        return DINOv3IMLOriginalLoss(loss_cfg)

    # keep existing losses below
```

如果项目没有统一 loss builder，而是在 training loop 中直接根据配置调用现有 loss，则加入：

```python
if str(cfg.get("loss", {}).get("name", "")) == "dinov3_iml_original":
    criterion = DINOv3IMLOriginalLoss(cfg.get("loss", {}))
    loss_dict = criterion(outputs, batch)
    loss = loss_dict["backward_loss"]
```

---

# Part D. 配置文件

## D1. 纯论文视频 baseline：`configs/b24_dinov3_iml_video_paperloss_lora32.yml`

这个配置用于确认 **原始 DINOv3-IML image baseline 迁移到 video frame 后的基础能力**。不加 CCM。

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
  name: "B24DINOv3IMLCCMVideoModel"

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
  # Strict paper-style target. Use only QKV first.
  targets: "attn.qkv"

tfcu:
  version: "ccm_predecoder"
  ccm:
    enabled: false
    dim: 128
    heads: 4
    q_resolution: 32
    kv_resolution: 16
    frame_mask: "lower_triangular"
    random_mask: {enabled: true, keep_prob: 0.70}
    fusion: "residual_concat"
    alpha_init: 0.002
    alpha_max: 0.05
    aux_head: false

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
  lr_ccm: 1.0e-4
  lr_decoder: 3.0e-4
  weight_decay: 5.0e-2

scheduler:
  type: "cosine"
  warmup_epochs: 5
  min_lr: 1.0e-6

train:
  n_epochs: 100
  save_dir: "runs/b24_dinov3_iml_video_paperloss_lora32"
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
  torchrun_log_dir: "runs/b24_dinov3_iml_video_paperloss_lora32/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
```

---

## D2. 推荐视频版：`configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml`

这个配置在 decoder 前加 CCM，但 loss 仍然严格保持原始论文 loss。

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
  name: "B24DINOv3IMLCCMVideoModel"

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

tfcu:
  version: "ccm_predecoder"
  ccm:
    enabled: true
    dim: 128
    heads: 4
    q_resolution: 32
    kv_resolution: 16
    frame_mask: "lower_triangular"
    random_mask: {enabled: true, keep_prob: 0.70}
    fusion: "residual_concat"
    alpha_init: 0.002
    alpha_max: 0.05
    aux_head: false

decoder:
  type: "dinov3_iml_head"
  in_channels: 1024
  hidden1: 512
  hidden2: 256
  # Keep BN for strict paper reproduction.
  # If video batch is unstable, switch to "gn" as an ablation.
  norm: "bn"
  image_size: 512

loss:
  name: "dinov3_iml_original"
  edge_lambda: 20.0
  edge_mask_width: 7

optimizer:
  learning_rate: 3.0e-4
  lr_lora: 3.0e-4
  lr_ccm: 1.0e-4
  lr_decoder: 3.0e-4
  weight_decay: 5.0e-2

scheduler:
  type: "cosine"
  warmup_epochs: 5
  min_lr: 1.0e-6

train:
  n_epochs: 100
  save_dir: "runs/b24_dinov3_iml_ccm_video_paperloss_lora32"
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
  torchrun_log_dir: "runs/b24_dinov3_iml_ccm_video_paperloss_lora32/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
```

---

# Part E. Optimizer 参数组注意事项

你项目已有类似：

```yaml
optimizer:
  learning_rate
  lr_lora
  lr_ccm
  lr_fgm
  lr_decoder
```

B24 里建议至少保证下面三组被正确识别：

| 参数名关键词 | 建议 lr |
|---|---:|
| `encoder.backbone.*lora*` | `3e-4` |
| `ccm` | `1e-4` |
| `decoder` / `seg_head` | `3e-4` |

如果当前 optimizer builder 不是按这些关键词分组，需要加上类似逻辑：

```python
if "lora_" in name:
    group = "lora"
elif name.startswith("ccm."):
    group = "ccm"
elif name.startswith("decoder."):
    group = "decoder"
else:
    group = "other"
```

第一轮不建议让 CCM 学得太快，所以：

```yaml
lr_ccm: 1.0e-4
```

而不是直接 `3e-4`。

---

# Part F. Smoke test

## F1. 新增 `tools/smoke_b24_dinov3_iml_ccm.py`

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
        default="configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml",
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
    print("aux keys:", out["aux"].keys())
    print("logits32:", tuple(out["aux"]["logits32"].shape))

    assert out["logits"].shape == (b, m, k, 1, h, w)
    assert torch.isfinite(out["logits"]).all()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total:,}, trainable={trainable:,}")

    for key in ["lora_", "ccm", "decoder"]:
        n = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and key in name)
        print(f"trainable[{key}] = {n:,}")

    print("B24 smoke test passed.")


if __name__ == "__main__":
    main()
```

运行：

```bash
python tools/smoke_b24_dinov3_iml_ccm.py   --config configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml
```

---

# Part G. Loss smoke test

可以单独验证原始 loss：

```python
import torch

from src.losses.dinov3_iml_original_loss import DINOv3IMLOriginalLoss

criterion = DINOv3IMLOriginalLoss({
    "edge_lambda": 20.0,
    "edge_mask_width": 7,
})

outputs = {
    "logits": torch.randn(1, 1, 4, 1, 512, 512),
}

batch = {
    "mask": (torch.rand(1, 1, 4, 1, 512, 512) > 0.95).float(),
}

loss_dict = criterion(outputs, batch)
print(loss_dict["loss"])
print(loss_dict.keys())
```

预期：

```text
loss 为有限值
包含 backward_loss / predict_loss / edge_loss / combined_loss
```

---

# Part H. 训练与 ablation 顺序

## H1. 第一组必须跑

```text
B24-A: DINOv3 + LoRA QKV + paper head + paper loss
B24-B: DINOv3 + LoRA QKV + CCM + paper head + paper loss
```

对比：

```text
B24-B - B24-A = CCM 对 video inpainting localization 的净贡献
```

不要第一轮就加 Dice、Tversky、mask128、boundary128，否则就无法判断论文结构和 CCM 的真实贡献。

---

## H2. 第二组建议跑

### 1. LoRA target ablation

严格论文：

```yaml
targets: "attn.qkv"
```

项目增强版：

```yaml
targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
```

解释：

- 如果严格论文 target 表现已经很好，保持简单。
- 如果视频任务明显欠拟合，再扩大 LoRA target。

### 2. Norm ablation

严格论文：

```yaml
decoder:
  norm: "bn"
```

视频稳定版：

```yaml
decoder:
  norm: "gn"
```

解释：

- BN 更贴近原始论文。
- GN 在小 batch 视频训练中可能更稳。

### 3. CCM alpha ablation

保守：

```yaml
alpha_init: 0.002
alpha_max: 0.05
```

更保守：

```yaml
alpha_init: 0.001
alpha_max: 0.02
```

解释：

- 如果 B24-B 比 B24-A 差，先降低 CCM 写入强度，而不是直接删 CCM。
- CCM 的目标是增强异常线索，不是大幅重写 DINO feature。

---

# Part I. 评估协议

建议每次都拆分报告：

```text
val_DVI
val_CPNET
test_DVI
test_CPNET
test_OPN
overall
```

重点看：

| 结果 | 解释 |
|---|---|
| B24-A 强，B24-B 更强 | CCM 对视频时序异常有效 |
| B24-A 强，B24-B 变差 | CCM 过度平滑或过拟合训练源 |
| B24-A 弱，B24-B 强 | 原论文 image head 不够，视频时序模块必要 |
| DVI/CPNET 强，OPN 差 | 学到了 inpainting method-specific artifact，泛化不足 |
| 所有方法都差 | loss/输入归一化/edge mask/训练 loop 可能有 bug |

---

# Part J. Agent 一次性修改指令

可以把下面这一段直接交给 coding agent：

```text
请在 My_model_v1 中新增 B24DINOv3IMLCCMVideoModel，用于复现 DINOv3-IML 的 frozen DINOv3 + LoRA QKV + 3-conv segmentation head，并支持在 decoder 前加入 CCM-Lite。

硬性要求：
1. 新增 src/models/decoders/dinov3_iml_head.py，实现 DINOv3IMLHead。
   - 输入 [N,1024,32,32]
   - Conv3x3 1024->512 + BN/ReLU
   - Conv3x3 512->256 + BN/ReLU
   - Conv1x1 256->1
   - bilinear upsample 到 512
   - 返回 logits，不做 sigmoid

2. 修改 src/models/decoders/__init__.py，导出 DINOv3IMLHead。

3. 新增 src/models/b24_dinov3_iml_ccm_video_model.py。
   - 使用 DINOv3B23Encoder
   - 支持输入 [B,M,K,3,H,W] 和 [B,K,3,H,W]
   - 输出 logits [B,M,K,1,H,W]
   - 如果 tfcu.ccm.enabled=true，则每个 clip 内调用 CCMLite
   - CCM 只作为 decoder 前 feature module，不新增 aux loss

4. 修改 src/models/__init__.py 和 src/models/builder.py。
   - 注册 B24DINOv3IMLCCMVideoModel

5. 新增 src/losses/dinov3_iml_original_loss.py。
   - 实现 DINOv3IMLOriginalLoss
   - loss = BCEWithLogits(logits, mask) + 20 * BCEWithLogits(logits, mask, weight=edge_mask)
   - edge_mask_width=7
   - 如果 batch 没有 edge_mask，则从 mask 用 dilation-erode 自动生成
   - 不使用 Dice/Tversky/mask128/boundary128/ccm_mask32 loss

6. 注册 loss：
   - loss.name == "dinov3_iml_original" 时使用 DINOv3IMLOriginalLoss

7. 新增两个配置：
   - configs/b24_dinov3_iml_video_paperloss_lora32.yml
   - configs/b24_dinov3_iml_ccm_video_paperloss_lora32.yml

8. 新增 smoke test：
   - tools/smoke_b24_dinov3_iml_ccm.py
   - 随机输入 [1,1,4,3,512,512]
   - 检查输出 [1,1,4,1,512,512]
   - 检查无 NaN
   - 打印 LoRA/CCM/decoder trainable 参数量
```

---

# Part K. 常见坑

## K1. 不要把 sigmoid 后的概率送进 BCEWithLogitsLoss

错误：

```python
prob = torch.sigmoid(logits)
loss = BCEWithLogitsLoss(prob, mask)
```

正确：

```python
loss = BCEWithLogitsLoss(logits, mask)
prob = torch.sigmoid(logits)  # only for visualization or metrics
```

---

## K2. 不要让 aux loss 偷偷生效

B24 paperloss 配置里不要出现：

```yaml
dice:
tversky:
boundary:
aux_loss:
  mask128:
  boundary128:
  ccm_mask32:
```

如果旧训练代码会默认读取这些字段，务必显式置零：

```yaml
loss:
  name: "dinov3_iml_original"
  edge_lambda: 20.0
  edge_mask_width: 7
  dice: {weight: 0.0}
  tversky: {weight: 0.0}
  boundary: {weight: 0.0}
  aux_loss:
    ccm_mask32: {enabled: false, weight: 0.0}
    mask128: {enabled: false, weight: 0.0}
    boundary128: {enabled: false, weight: 0.0}
```

---

## K3. 如果显存不足

优先级从高到低：

1. `encoder_chunk: 1`
2. `use_activation_checkpoint: true`
3. `grad_accum_steps` 增大
4. `num_clips: 2, num_frames: 4`
5. 最后才考虑把 decoder 输入从 1024 降到 128

如果把 decoder 输入降到 128，就不再是严格论文 head，只能叫工程版：

```text
DINOv3 feature -> CCM -> projection 1024->128 -> 3-conv head
```

第一轮不推荐这么做。

---

## K4. 如果 CCM 版更差

不要马上否定 CCM，先按顺序查：

1. `tfcu.ccm.aux_head` 是否为 false 或 aux loss 是否为 0。
2. `alpha_init` 是否过大。
3. `alpha_max` 是否过大。
4. `lr_ccm` 是否过大。
5. `targets` 是否从 `attn.qkv` 扩到了太多层。
6. 验证是否只用了很少 batch。
7. OPN 是否单独崩；如果只有 OPN 崩，是泛化问题，不一定是训练 bug。

---

# Part L. 推荐最终路线

第一阶段：

```text
B24-A no CCM, paper loss
B24-B CCM pre-decoder, paper loss
```

第二阶段：

```text
B24-B + GN decoder
B24-B + lower CCM alpha
B24-B + expanded LoRA targets
```

第三阶段：

```text
回到旧 b23_ccm_lite_lora32，与 B24-B 对比：
- 如果 B24-B 更好：原论文 head + paper loss 是更强 baseline。
- 如果旧 B23 更好：LiteBoundaryDecoder 的多尺度上采样和 aux loss 对视频 inpainting 更关键。
- 如果 B24-A 和 B24-B 都强：说明 DINOv3 dense feature 本身已经足够强，后续重点应转向数据泛化和评估协议。
```

---

# References

- DINOv3-IML paper: `DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics`
- Official code: `Irennnne/DINOv3-IML`
- Your project: `azaaa7/My_model_v1`
- Current strong config: `configs/b23_ccm_lite_lora32.yml`
