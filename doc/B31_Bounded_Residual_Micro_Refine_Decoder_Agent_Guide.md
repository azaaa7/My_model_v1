# B31: Bounded Residual Micro-Refine Decoder 实现文档

> 目标：在当前最好的 **DINOv3 + No-Gate 3D Conv / StA 视频模型** 上，新增一个小而精的 decoder refinement 模块，尽可能提升域内 DVI/CPNET 表现，同时保持或小幅提升域外 OPN 表现。  
> 核心约束：**不引入大模型，不引入 SAM2/轨迹估计，不引入可学习门控，不重写 DINOv3 特征。**

---

## 0. 当前问题判断

当前最好版本已经证明：

```text
DINOv3 frozen + LoRA
  + No-Gate 3D Conv / StA
  + DINOv3-IML paper-style head
  + BCE + edge-aware BCE
```

比 CCM 更稳。你的当前主要问题不是域外，而是：

```text
1. 域外 OPN 已经强，不能为了域内提升把泛化破坏掉；
2. 域内 DVI/CPNET 还没达到 SOTA；
3. 当前 DINOv3-IML head 只在 32×32 上输出 logits，然后直接 bilinear 到 512；
4. 因此域内短板大概率来自 mask 细节、边界、薄结构、小区域和局部召回。
```

所以新增模块应该只做 **logit-level micro refinement**，而不是 feature-level heavy fusion。

---

## 1. 模块名称

新增 decoder：

```text
BoundedResidualMicroRefineDecoder
```

建议模型版本名：

```text
B31DINOv3IMLNoGateStABRMRVideoModel
```

其中 BRMR = **Bounded Residual Micro-Refine**。

---

## 2. 设计原则

### 2.1 不做的事

不要做：

```text
- 不加 SAM2；
- 不加 tracker；
- 不加 cross-attention；
- 不加 CCM；
- 不加 learned gate；
- 不在 512×512 上做大 decoder；
- 不直接大幅改写 DINOv3 feature。
```

### 2.2 要做的事

只做：

```text
coarse logits32
  → bounded residual delta32
  → optional bounded residual delta128
  → final logits512
```

其中 residual 被严格限制幅度：

```python
delta = clip_value * torch.tanh(raw_delta / clip_value)
```

这样即使 residual head 过拟合域内，也不能无限重写 base prediction，有利于保持 OPN 泛化。

---

## 3. 结构图

当前最好 3D conv 版本大致是：

```text
[B,M,K,3,512,512]
    ↓
DINOv3B23Encoder + LoRA
    ↓
No-Gate 3D Conv / StA Adapter
    ↓
DINOv3IMLHead
    ↓
logits32 → bilinear → logits512
```

改成：

```text
[B,M,K,3,512,512]
    ↓
DINOv3B23Encoder + LoRA
    ↓
No-Gate 3D Conv / StA Adapter
    ↓
Base 3-conv head
    ↓
coarse logits32
    ↓
Low-res 3D bounded residual refine
    ↓
refined logits32
    ↓
Upsample to 128
    ↓
Tiny 2D boundary/detail residual refine
    ↓
refined logits128
    ↓
Bilinear to 512
```

最终输出仍是：

```python
out["logits"]     # [B,M,K,1,H,W]
out["aux"]        # 包含 logits32, logits128, delta32, delta128, debug
```

---

## 4. 为什么这个模块适合当前目标

这个模块的作用不是增强泛化，而是**受控地补域内细节**：

| 子模块 | 作用 | 域外风险 |
|---|---|---|
| low-res 3D residual | 让 coarse mask 在时间上更稳定、补域内召回 | 低，residual bounded + zero-init |
| 128×128 micro detail | 修复 32→512 直接插值导致的边界粗糙 | 低，只修 logit，不改 feature |
| residual clipping | 限制域内增强幅度 | 很低 |
| residual L1 optional | 进一步防止 OOD 崩 | 很低 |

---

## 5. 新增文件

新增：

```text
src/models/decoders/bounded_residual_micro_refine_decoder.py
```

修改：

```text
src/models/decoders/__init__.py
src/models/builder.py 或对应 build_model 文件
当前最好 3D Conv / NoGateStA 模型文件
optimizer 参数分组文件，可选
loss 文件，可选
configs/b31_dinov3_iml_nogate_sta_brmr_lora32.yml
```

如果当前项目里你的最好 3D conv 模型文件名不是上面名字，让 agent 搜索：

```text
DINOv3IMLHead
NoGate
Spatiotemporal
Conv3d
DINOv3B23Encoder
```

找到当前最好模型后，在它的 decoder 调用处接入 BRMR。

---

## 6. 新增 decoder 代码

创建文件：

```text
src/models/decoders/bounded_residual_micro_refine_decoder.py
```

内容如下：

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_iml_head import DINOv3IMLHead


def _make_norm2d(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 2D norm: {norm}")


def _make_norm3d(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D norm: {norm}")


def _zero_init_conv(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _bounded_delta(raw_delta: torch.Tensor, clip_value: float) -> torch.Tensor:
    if clip_value <= 0:
        return raw_delta
    return float(clip_value) * torch.tanh(raw_delta / float(clip_value))


class BoundedResidualMicroRefineDecoder(nn.Module):
    """
    Paper-style DINOv3-IML head + bounded logit residual refinement.

    Designed for video inpainting localization:
      1) Base head produces coarse logits32.
      2) A tiny 3D residual head refines logits32 using local temporal context.
      3) A tiny 2D residual head refines logits128 for boundary/detail.
      4) Residuals are bounded and final residual layers are zero-initialized.

    This module intentionally has no learned gate and no attention.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}

        self.in_channels = int(cfg.get("in_channels", 1024))
        self.image_size = int(cfg.get("image_size", cfg.get("input_size", 512)))
        self.norm = str(cfg.get("norm", "gn"))

        # Base paper-style head.
        base_cfg = dict(cfg)
        base_cfg.setdefault("type", "dinov3_iml_head")
        base_cfg.setdefault("in_channels", self.in_channels)
        base_cfg.setdefault("hidden1", int(cfg.get("hidden1", self.in_channels // 2)))
        base_cfg.setdefault("hidden2", int(cfg.get("hidden2", self.in_channels // 4)))
        base_cfg.setdefault("image_size", self.image_size)
        base_cfg.setdefault("norm", self.norm)
        self.base_head = DINOv3IMLHead(base_cfg)

        mr_cfg = cfg.get("micro_refine", {}) or {}
        self.enabled = bool(mr_cfg.get("enabled", True))
        self.refine_channels = int(mr_cfg.get("channels", 64))
        self.high_res = int(mr_cfg.get("high_res", 128))
        self.delta32_clip = float(mr_cfg.get("delta32_clip", 1.0))
        self.delta128_clip = float(mr_cfg.get("delta128_clip", 0.75))
        self.use_high128 = bool(mr_cfg.get("use_high128", True))
        self.use_prob = bool(mr_cfg.get("use_prob", True))
        self.use_uncertainty = bool(mr_cfg.get("use_uncertainty", True))
        self.detach_coarse_for_refine = bool(mr_cfg.get("detach_coarse_for_refine", False))

        # Project DINO feature once. Keep this small.
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.refine_channels, kernel_size=1, bias=False),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
        )

        low_in = self.refine_channels + 1
        if self.use_prob:
            low_in += 1
        if self.use_uncertainty:
            low_in += 1

        # Low-res 3D refinement on [B, C, K, 32, 32].
        # Depthwise 3D conv keeps it cheap.
        self.low3d_pre = nn.Sequential(
            nn.Conv3d(low_in, self.refine_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, self.refine_channels),
            nn.GELU(),
        )
        self.low3d_dw = nn.Sequential(
            nn.Conv3d(
                self.refine_channels,
                self.refine_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                groups=self.refine_channels,
                bias=False,
            ),
            _make_norm3d(self.norm, self.refine_channels),
            nn.GELU(),
        )
        self.low3d_out = nn.Conv3d(self.refine_channels, 1, kernel_size=1, bias=True)
        _zero_init_conv(self.low3d_out)

        # Optional 128x128 detail refinement.
        high_in = self.refine_channels + 1
        if self.use_prob:
            high_in += 1
        if self.use_uncertainty:
            high_in += 1

        self.high2d = nn.Sequential(
            nn.Conv2d(high_in, self.refine_channels, kernel_size=3, padding=1, bias=False),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
            nn.Conv2d(
                self.refine_channels,
                self.refine_channels,
                kernel_size=3,
                padding=1,
                groups=self.refine_channels,
                bias=False,
            ),
            _make_norm2d(self.norm, self.refine_channels),
            nn.GELU(),
            nn.Conv2d(self.refine_channels, 1, kernel_size=1, bias=True),
        )
        _zero_init_conv(self.high2d[-1])

    @staticmethod
    def _logit_cues(logits: torch.Tensor, use_prob: bool, use_uncertainty: bool) -> list[torch.Tensor]:
        cues = [logits]
        prob = torch.sigmoid(logits)
        if use_prob:
            cues.append(prob)
        if use_uncertainty:
            # Max at decision boundary, small at confident fg/bg.
            cues.append(4.0 * prob * (1.0 - prob))
        return cues

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Fallback image-style forward. Treat N as B*K with K=1."""
        if features.ndim != 4:
            raise ValueError(f"features must be [N,C,H,W], got {tuple(features.shape)}")
        n, c, h, w = features.shape
        video_features = features.reshape(n, 1, c, h, w)
        out = self.forward_video(video_features, output_size=output_size)
        # Convert [N,1,...] outputs back to [N,...]
        return {
            "logits": out["logits"].reshape(n, 1, *out["logits"].shape[-2:]),
            "logits32": out["logits32"].reshape(n, 1, *out["logits32"].shape[-2:]),
            "logits32_coarse": out["logits32_coarse"].reshape(n, 1, *out["logits32_coarse"].shape[-2:]),
            "logits128": out["logits128"].reshape(n, 1, *out["logits128"].shape[-2:]),
            "delta32": out["delta32"].reshape(n, 1, *out["delta32"].shape[-2:]),
            "delta128": out["delta128"].reshape(n, 1, *out["delta128"].shape[-2:]),
            "debug": out["debug"],
        }

    def forward_video(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """
        Args:
            features: [B,K,C,32,32]
            output_size: final logit size, usually (512,512)

        Returns:
            logits: [B,K,1,H,W]
            logits32: [B,K,1,32,32]
            logits32_coarse: [B,K,1,32,32]
            logits128: [B,K,1,128,128]
            delta32: [B,K,1,32,32]
            delta128: [B,K,1,128,128]
        """
        if features.ndim != 5:
            raise ValueError(f"features must be [B,K,C,H,W], got {tuple(features.shape)}")
        if output_size is None:
            output_size = (self.image_size, self.image_size)

        b, k, c, fh, fw = features.shape
        flat_features = features.reshape(b * k, c, fh, fw)

        base = self.base_head(flat_features, output_size=(fh, fw))
        logits32_coarse = base["logits32"].reshape(b, k, 1, fh, fw)

        if not self.enabled:
            logits32 = logits32_coarse
            logits128 = F.interpolate(
                logits32.reshape(b * k, 1, fh, fw),
                size=(self.high_res, self.high_res),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, k, 1, self.high_res, self.high_res)
            logits = F.interpolate(
                logits128.reshape(b * k, 1, self.high_res, self.high_res),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(b, k, 1, *output_size)
            zero32 = torch.zeros_like(logits32)
            zero128 = torch.zeros_like(logits128)
            return {
                "logits": logits,
                "logits32": logits32,
                "logits32_coarse": logits32_coarse,
                "logits128": logits128,
                "delta32": zero32,
                "delta128": zero128,
                "debug": {
                    "decoder_type": "BoundedResidualMicroRefineDecoder",
                    "micro_refine_enabled": False,
                    "feature_shape": tuple(features.shape),
                },
            }

        feat32 = self.feat_proj(flat_features).reshape(b, k, self.refine_channels, fh, fw)

        coarse_for_refine = logits32_coarse.detach() if self.detach_coarse_for_refine else logits32_coarse
        low_cues = self._logit_cues(coarse_for_refine, self.use_prob, self.use_uncertainty)
        low_in = torch.cat([feat32, *low_cues], dim=2)  # [B,K,Cin,H,W]
        low_in = low_in.permute(0, 2, 1, 3, 4).contiguous()  # [B,Cin,K,H,W]

        raw_delta32 = self.low3d_out(self.low3d_dw(self.low3d_pre(low_in)))
        delta32 = _bounded_delta(raw_delta32, self.delta32_clip).permute(0, 2, 1, 3, 4).contiguous()
        logits32 = logits32_coarse + delta32

        # 128 detail refinement.
        logits128_base = F.interpolate(
            logits32.reshape(b * k, 1, fh, fw),
            size=(self.high_res, self.high_res),
            mode="bilinear",
            align_corners=False,
        )

        if self.use_high128:
            feat128 = F.interpolate(
                feat32.reshape(b * k, self.refine_channels, fh, fw),
                size=(self.high_res, self.high_res),
                mode="bilinear",
                align_corners=False,
            )
            high_cues = self._logit_cues(logits128_base, self.use_prob, self.use_uncertainty)
            high_in = torch.cat([feat128, *high_cues], dim=1)
            raw_delta128 = self.high2d(high_in)
            delta128_flat = _bounded_delta(raw_delta128, self.delta128_clip)
            logits128_flat = logits128_base + delta128_flat
        else:
            delta128_flat = torch.zeros_like(logits128_base)
            logits128_flat = logits128_base

        logits = F.interpolate(
            logits128_flat,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, k, 1, *output_size)

        logits128 = logits128_flat.reshape(b, k, 1, self.high_res, self.high_res)
        delta128 = delta128_flat.reshape(b, k, 1, self.high_res, self.high_res)

        debug = {
            "decoder_type": "BoundedResidualMicroRefineDecoder",
            "micro_refine_enabled": True,
            "feature_shape": tuple(features.shape),
            "logits32_coarse_shape": tuple(logits32_coarse.shape),
            "logits32_shape": tuple(logits32.shape),
            "logits128_shape": tuple(logits128.shape),
            "logits_shape": tuple(logits.shape),
            "delta32_clip": self.delta32_clip,
            "delta128_clip": self.delta128_clip,
            "delta32_abs_mean": float(delta32.detach().abs().mean().cpu()),
            "delta128_abs_mean": float(delta128.detach().abs().mean().cpu()),
            "delta32_abs_max": float(delta32.detach().abs().max().cpu()),
            "delta128_abs_max": float(delta128.detach().abs().max().cpu()),
        }

        return {
            "logits": logits,
            "logits32": logits32,
            "logits32_coarse": logits32_coarse,
            "logits128": logits128,
            "delta32": delta32,
            "delta128": delta128,
            "debug": debug,
        }
```

---

## 7. 导出 decoder

修改：

```text
src/models/decoders/__init__.py
```

加入：

```python
from .bounded_residual_micro_refine_decoder import BoundedResidualMicroRefineDecoder
```

如果项目里有 `build_decoder()`，加入：

```python
if decoder_type in {"bounded_residual_micro_refine", "brmr"}:
    return BoundedResidualMicroRefineDecoder(cfg)
```

如果没有 decoder builder，而是在模型里直接 import，则在当前最好 3D conv 模型里直接：

```python
from .decoders import DINOv3IMLHead, BoundedResidualMicroRefineDecoder
```

---

## 8. 接入当前最好 3D Conv / No-Gate StA 模型

找到当前最好模型中类似下面的代码：

```python
dec = self.decoder(x_dec.reshape(b * k, feat_c, feat_h, feat_w), output_size=(h, w))
logits_list.append(dec["logits"].reshape(b, k, 1, h, w))
logits32_list.append(dec["logits32"].reshape(b, k, 1, feat_h, feat_w))
```

替换为：

```python
if hasattr(self.decoder, "forward_video"):
    dec = self.decoder.forward_video(x_dec, output_size=(h, w))
    logits_list.append(dec["logits"])
    logits32_list.append(dec["logits32"])

    if "logits32_coarse" in dec:
        logits32_coarse_list.append(dec["logits32_coarse"])
    if "logits128" in dec:
        logits128_list.append(dec["logits128"])
    if "delta32" in dec:
        delta32_list.append(dec["delta32"])
    if "delta128" in dec:
        delta128_list.append(dec["delta128"])
else:
    dec = self.decoder(x_dec.reshape(b * k, feat_c, feat_h, feat_w), output_size=(h, w))
    logits_list.append(dec["logits"].reshape(b, k, 1, h, w))
    logits32_list.append(dec["logits32"].reshape(b, k, 1, feat_h, feat_w))
```

在循环前初始化：

```python
logits32_coarse_list = []
logits128_list = []
delta32_list = []
delta128_list = []
```

在 forward 末尾构建 aux：

```python
aux = {
    "logits32": logits32,
    "debug": debug,
}

if logits32_coarse_list:
    aux["logits32_coarse"] = torch.stack(logits32_coarse_list, dim=1)
if logits128_list:
    aux["logits128"] = torch.stack(logits128_list, dim=1)
if delta32_list:
    aux["delta32"] = torch.stack(delta32_list, dim=1)
if delta128_list:
    aux["delta128"] = torch.stack(delta128_list, dim=1)
```

同时 debug 中加入：

```python
debug[f"clip{clip_idx}_decoder"] = dec["debug"]
```

---

## 9. decoder 构建逻辑

如果当前模型原来写的是：

```python
self.decoder = DINOv3IMLHead(decoder_cfg)
```

改成：

```python
decoder_type = str(decoder_cfg.get("type", "dinov3_iml_head")).lower()
if decoder_type in {"bounded_residual_micro_refine", "brmr"}:
    self.decoder = BoundedResidualMicroRefineDecoder(decoder_cfg)
else:
    self.decoder = DINOv3IMLHead(decoder_cfg)
```

注意：BRMR 的输入仍然是 1024 通道 DINO/StA feature，不要投影到 128 再送 decoder，除非你的当前最好 3D conv 模型本来就是 128 通道。

---

## 10. 推荐配置

新增：

```text
configs/b31_dinov3_iml_nogate_sta_brmr_lora32.yml
```

以当前最好 3D conv 配置为底座，只改 decoder 和训练策略。

```yaml
model:
  name: B25DINOv3IMLNoGateStAVideoModel   # 如果已有 B25 名字不同，用当前最好模型名

input_size: 512
encoder_chunk: 8
use_activation_checkpoint: true

sampling:
  num_clips: 4
  num_frames: 4

# 保持你当前最好的 No-Gate 3D Conv / StA 配置，不要改大。
sta:
  enabled: true
  bottleneck_dim: 128
  spatial_kernel: [1, 3, 3]
  temporal_kernel: [3, 1, 1]
  fusion: fixed_average
  zero_init_residual: true

# 关键新增：BRMR decoder
decoder:
  type: bounded_residual_micro_refine
  in_channels: 1024
  hidden1: 512
  hidden2: 256
  image_size: 512
  norm: gn

  micro_refine:
    enabled: true
    channels: 64
    high_res: 128
    use_high128: true

    # Bounded residual: 先保守，防止 OPN 下降。
    delta32_clip: 1.0
    delta128_clip: 0.75

    use_prob: true
    use_uncertainty: true
    detach_coarse_for_refine: false

loss:
  name: dinov3_iml_original
  edge_lambda: 20.0
  edge_mask_width: 7

  # 可选，第一轮可以先关。
  residual_regularization:
    enabled: false
    delta32_l1_weight: 0.003
    delta128_l1_weight: 0.001

optimizer:
  learning_rate: 3.0e-4
  lr_lora: 1.0e-4
  lr_sta: 1.0e-4
  lr_decoder: 3.0e-4
  lr_micro_refine: 3.0e-4
  weight_decay: 5.0e-2

training:
  # 如果从当前最好 3D conv checkpoint 微调，推荐两阶段。
  init_from: path/to/current_best_nogate_sta_checkpoint.pth

  stage1:
    epochs: 5
    freeze_encoder: true
    freeze_lora: true
    freeze_sta: true
    train_decoder: true
    train_micro_refine: true

  stage2:
    epochs: 25
    freeze_encoder: true
    freeze_lora: false
    freeze_sta: false
    train_decoder: true
    train_micro_refine: true
    lora_lr_scale: 0.3
    sta_lr_scale: 0.5

validation:
  val_full_video: true
  max_batches: null
```

---

## 11. optimizer 参数组建议

如果项目 optimizer 里按模块名分组，加入：

```python
if "micro_refine" in name or "low3d" in name or "high2d" in name:
    group = "micro_refine"
elif "decoder" in name:
    group = "decoder"
elif "sta" in name or "spatiotemporal" in name or "adapter" in name:
    group = "sta"
elif "lora" in name.lower():
    group = "lora"
```

学习率建议：

```text
micro_refine: 3e-4
base decoder : 3e-4
sta          : 1e-4 或 5e-5
LoRA         : 1e-4 或 3e-5
```

如果从当前 best checkpoint 微调，最稳的是：

```text
先训 micro_refine + decoder，冻结 LoRA/StA；
然后低学习率解冻 LoRA/StA。
```

---

## 12. optional residual regularization

第一轮可以不加新 loss，只用 final logits 的原始 paper loss。

如果发现 DVI/CPNET 涨但 OPN 掉，打开 residual regularization。

在 loss 里加：

```python
aux = outputs.get("aux", {})
res_loss = 0.0
if "delta32" in aux:
    res_loss = res_loss + delta32_l1_weight * aux["delta32"].abs().mean()
if "delta128" in aux:
    res_loss = res_loss + delta128_l1_weight * aux["delta128"].abs().mean()
loss = loss + res_loss
```

建议配置：

```yaml
loss:
  residual_regularization:
    enabled: true
    delta32_l1_weight: 0.003
    delta128_l1_weight: 0.001
```

不要一开始就加 Dice/Tversky/大量 aux loss，否则不好判断 BRMR 本身是否有效。

---

## 13. 训练建议

### 13.1 第一组实验

```text
E0: 当前最好 3D conv 模型
E1: E0 + BRMR，no residual regularization
E2: E0 + BRMR，residual L1 on
E3: E0 + BRMR，但 use_high128=false
E4: E0 + BRMR，delta32_clip=0.5, delta128_clip=0.5
```

### 13.2 判断标准

主目标：

```text
same_source_avg IoU / F1 提升
```

约束：

```text
OPN IoU 不低于当前最好值 - 0.003
OPN Precision 保持 > 0.90
```

辅助观察：

```text
delta32_abs_mean 不应长期 > 0.25
delta128_abs_mean 不应长期 > 0.20
delta32_abs_max 应接近但不频繁撞 clip
delta128_abs_max 应接近但不频繁撞 clip
```

如果 delta 大量撞到 clip，说明 residual head 过强，应降低 clip 或打开 L1。

---

## 14. smoke test

新增脚本或临时测试：

```python
import torch
from src.models.decoders import BoundedResidualMicroRefineDecoder

cfg = {
    "type": "bounded_residual_micro_refine",
    "in_channels": 1024,
    "hidden1": 512,
    "hidden2": 256,
    "image_size": 512,
    "norm": "gn",
    "micro_refine": {
        "enabled": True,
        "channels": 64,
        "high_res": 128,
        "delta32_clip": 1.0,
        "delta128_clip": 0.75,
        "use_high128": True,
        "use_prob": True,
        "use_uncertainty": True,
    },
}

model = BoundedResidualMicroRefineDecoder(cfg).cuda().train()
x = torch.randn(1, 4, 1024, 32, 32, device="cuda")
out = model.forward_video(x, output_size=(512, 512))

assert out["logits"].shape == (1, 4, 1, 512, 512)
assert out["logits32"].shape == (1, 4, 1, 32, 32)
assert out["logits32_coarse"].shape == (1, 4, 1, 32, 32)
assert out["logits128"].shape == (1, 4, 1, 128, 128)
assert out["delta32"].shape == (1, 4, 1, 32, 32)
assert out["delta128"].shape == (1, 4, 1, 128, 128)
assert torch.isfinite(out["logits"]).all()

# zero-init check: initial deltas should be near zero
print(out["delta32"].abs().mean().item(), out["delta128"].abs().mean().item())
```

预期：

```text
delta32 mean ≈ 0
delta128 mean ≈ 0
```

---

## 15. 实现注意事项

1. **BRMR 必须接在当前最好 3D conv adapter 后面。**  
   不要回退到 B24 CCM 版本。

2. **不要把 BRMR 的 high128 做太大。**  
   128 足够，先不要上 256，否则显存和过拟合风险都上升。

3. **不要引入 learned gate。**  
   bounded residual 已经提供控制能力。

4. **不要改变主 loss。**  
   第一轮仍然使用原始 paper loss：BCE + edge-aware BCE。

5. **优先用 GroupNorm。**  
   视频 batch 小，BatchNorm 容易造成 calibration 波动。

6. **务必保存 debug。**  
   重点看 delta32/delta128 的 mean 和 max。

7. **如果域外掉，先调小 residual，不要马上删模块。**  
   调参顺序：

```text
打开 residual L1
↓
delta128_clip: 0.75 → 0.5
↓
delta32_clip: 1.0 → 0.5
↓
use_high128: true → false
```

---

## 16. 预期收益

理想变化：

```text
DVI_20 IoU   +0.5 ~ +1.5
CPNET_20 IoU +0.5 ~ +1.5
OPN_20 IoU   -0.3 ~ +0.5 内波动
```

如果只看 average，可能提升不大；但如果看域内 SOTA 差距、boundary F1、小区域 recall，应该更明显。

---

## 17. agent 最短任务说明

```text
在当前最好的 DINOv3 + No-Gate 3D Conv / StA 视频模型上新增 Bounded Residual Micro-Refine Decoder。

必须完成：
1. 新增 src/models/decoders/bounded_residual_micro_refine_decoder.py。
2. 在 decoders/__init__.py 导出 BoundedResidualMicroRefineDecoder。
3. 修改当前最好 3D conv 模型，使其支持 decoder.forward_video(x_clip, output_size=(h,w))。
4. decoder.type="bounded_residual_micro_refine" 时使用 BoundedResidualMicroRefineDecoder，否则保持原 DINOv3IMLHead。
5. forward 输出 aux["logits32"], aux["logits32_coarse"], aux["logits128"], aux["delta32"], aux["delta128"], aux["debug"]。
6. 新增 configs/b31_dinov3_iml_nogate_sta_brmr_lora32.yml，以当前最好 3D conv 配置为底座，只替换 decoder。
7. 第一轮训练不要新增 Dice/Tversky/aux mask loss，继续用 BCE + edge-aware BCE。
8. 可选增加 residual L1，但默认关闭。
9. 做 smoke test：B=1,K=4,C=1024,H=W=32，输出 logits [1,4,1,512,512]，无 NaN，初始 delta 接近 0。
```

---

## 18. 一句话总结

BRMR 的目的不是做一个更强的大 decoder，而是：

```text
在当前强泛化 3D conv baseline 上，增加一个受边界约束、受幅度约束、zero-init 的小 residual decoder，专门补域内 mask 细节，同时尽量不破坏 OPN 泛化。
```
