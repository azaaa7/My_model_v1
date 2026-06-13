# Agent 实现指南：直接实现 VidEoMT-style 最终版 Window Query Fusion 模型

## 0. 本次任务目标

本次不要先做逐步实验版，也不要实现一堆 ablation 分支。  
目标是 **直接实现最终版模型**：

```text
DINOv3-B23
+ Window input [B, W, T, 3, H, W]
+ Learned forgery queries
+ VidEoMT-style query propagation / query fusion
+ Bidirectional forward/backward average
+ LiteBoundaryDecoder
+ BCE + Dice loss
- CCM
- FGM
- FGM bank
- gate_f / gate_b
- sigmoid MLP gate
- RelayFormer slicing
- GLR token
- query consistency loss
```

核心思想：

```text
用 VidEoMT 风格的 query propagation / query fusion
替代原项目中的 CCM 和 FGM。
```

本次实现应尽量简单、稳定、可训练。  
不要为了“更强”而加入新的复杂模块。

---

## 1. 最终模型名称

新增模型：

```text
B23VideoMTWindowModel
```

建议文件：

```text
src/models/b23_videomt_window_model.py
```

新增 query fusion 模块目录：

```text
src/models/videomt/
```

新增文件：

```text
src/models/videomt/__init__.py
src/models/videomt/window_query_fusion.py
```

新增 loss 文件：

```text
src/losses/videomt_loss.py
```

新增配置：

```text
configs/b23_videomt_window.yaml
```

---

## 2. 输入输出规范

### 2.1 模型输入

统一使用 window 输入：

```python
video: Tensor[B, W, T, 3, H, W]
```

其中：

```text
B = batch size
W = number of temporal windows
T = frames per window
H, W = 当前 resize 后输入尺寸，例如 512 × 512
```

注意变量命名时不要把 width 也写成 `W`，代码中建议：

```python
B, num_windows, num_frames, C, H, W_img = video.shape
```

### 2.2 兼容旧输入

如果输入是：

```python
[B, T, 3, H, W]
```

则自动扩展为单 window：

```python
if video.ndim == 5:
    video = video[:, None]
```

### 2.3 模型输出

输出必须保持：

```python
outputs = {
    "logits": logits,      # [B, W, T, 1, H, W]
    "aux": aux,
}
```

其中：

```python
logits.shape == [B, num_windows, num_frames, 1, H, W_img]
```

---

## 3. 不要实现的内容

本次最终版不要实现以下内容：

```text
1. CCM
2. FGM
3. FGM bank
4. quality gate
5. gate_f / gate_b
6. sigmoid MLP gate
7. query consistency loss
8. temporal smoothness loss
9. contrastive / re-id loss
10. RelayFormer 原分辨率切片
11. GLR tokens
12. 4D RoPE
13. optical flow loss
14. bank consistency loss
```

特别禁止以下形式：

```python
gate_f = sigmoid(MLP([q_lrn, q_f]))
gate_b = sigmoid(MLP([q_lrn, q_b]))
q_t = q_lrn + gate_f * q_f + gate_b * q_b
```

本次只允许 VidEoMT 风格的简单 fusion：

```python
q_in = Linear(prev_q_out) + q_lrn
```

---

## 4. 最终 Query Fusion 公式

### 4.1 Forward direction

```text
Q_f_in[0] = Q_lrn
Q_f_in[t] = Linear_f(Q_f_out[t-1]) + Q_lrn
```

### 4.2 Backward direction

```text
Q_b_in[T-1] = Q_lrn
Q_b_in[t] = Linear_b(Q_b_out[t+1]) + Q_lrn
```

### 4.3 Bidirectional fusion

```text
Q_out[t] = 0.5 * (Q_f_out[t] + Q_b_out[t])
```

不要加 gate。  
不要加可学习加权。  
不要加 quality score。  
第一版就用平均。

---

## 5. WindowQueryFusion 模块设计

文件：

```text
src/models/videomt/window_query_fusion.py
```

### 5.1 输入输出

输入：

```python
features: Tensor[B, T, C, Hf, Wf]
```

来自 DINOv3-B23，例如：

```python
[B, T, 1024, 32, 32]
```

输出：

```python
enhanced_features: Tensor[B, T, C, Hf, Wf]
aux: dict
```

aux 至少包含：

```python
aux = {
    "query_states": query_states,   # [B, T, Nq, C]
}
```

### 5.2 模块结构

实现两个类：

```python
class QueryPatchBlock(nn.Module):
    ...
```

```python
class WindowQueryFusion(nn.Module):
    ...
```

---

## 6. QueryPatchBlock 实现要求

`QueryPatchBlock` 用于让 forgery queries 与当前帧 patch tokens 交互。

### 6.1 输入

```python
queries: Tensor[B, Nq, C]
patch_tokens: Tensor[B, N, C]
```

其中：

```text
Nq = number of forgery queries
N = Hf * Wf
C = DINOv3 feature dim，默认 1024
```

### 6.2 输出

```python
updated_queries: Tensor[B, Nq, C]
```

### 6.3 推荐实现

```python
class QueryPatchBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=8, ffn_ratio=4.0, dropout=0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden = int(dim * ffn_ratio)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, patch_tokens):
        q = queries

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(patch_tokens)
        cross, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=False,
        )
        q = q + cross

        q_norm = self.self_norm(q)
        self_out, _ = self.self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm,
            need_weights=False,
        )
        q = q + self_out

        q = q + self.ffn(self.ffn_norm(q))
        return q
```

注意：

```text
1. 不要 gate。
2. 不要 sigmoid。
3. 不要 bank。
4. 不要 query consistency loss。
```

---

## 7. WindowQueryFusion 实现要求

### 7.1 初始化参数

```python
class WindowQueryFusion(nn.Module):
    def __init__(
        self,
        dim=1024,
        num_queries=16,
        num_heads=8,
        ffn_ratio=4.0,
        dropout=0.0,
        bidirectional=True,
        residual_alpha_init=0.0,
    ):
        ...
```

### 7.2 成员变量

```python
self.learned_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

self.forward_linear = nn.Linear(dim, dim)
self.backward_linear = nn.Linear(dim, dim)

self.forward_block = QueryPatchBlock(...)
self.backward_block = QueryPatchBlock(...)

self.query_to_feature = nn.Linear(dim, dim)
self.residual_alpha = nn.Parameter(torch.tensor(residual_alpha_init))
```

### 7.3 为什么需要 residual_alpha

把 query context 加回 feature 时，建议使用：

```python
enhanced = features + alpha * context
```

其中：

```python
alpha = self.residual_alpha
```

默认：

```text
residual_alpha_init = 0.0
```

这不是 gate。  
它是训练稳定用的 residual scale。  
初始为 0 时，模型从 DINOv3 静态 baseline 开始，不会被随机 query 破坏。

---

## 8. WindowQueryFusion forward 伪代码

```python
def forward(self, features):
    """
    features: [B, T, C, Hf, Wf]
    """
    B, T, C, Hf, Wf = features.shape

    patch_tokens = features.flatten(3).transpose(2, 3)
    # [B, T, N, C]

    q_lrn = self.learned_queries[None].expand(B, -1, -1)
    # [B, Nq, C]

    q_forward = self._run_forward(patch_tokens, q_lrn)

    if self.bidirectional:
        q_backward = self._run_backward(patch_tokens, q_lrn)
        q_states = 0.5 * (q_forward + q_backward)
    else:
        q_states = q_forward

    query_context = q_states.mean(dim=2)
    # [B, T, C]

    query_context = self.query_to_feature(query_context)
    query_context = query_context[:, :, :, None, None]

    enhanced = features + self.residual_alpha * query_context

    aux = {
        "query_states": q_states,
    }

    return enhanced, aux
```

---

## 9. Forward propagation 实现

```python
def _run_forward(self, patch_tokens, q_lrn):
    """
    patch_tokens: [B, T, N, C]
    q_lrn: [B, Nq, C]
    """
    B, T, N, C = patch_tokens.shape
    outputs = []
    prev = None

    for t in range(T):
        if prev is None:
            q_in = q_lrn
        else:
            q_in = self.forward_linear(prev) + q_lrn

        q_out = self.forward_block(q_in, patch_tokens[:, t])
        outputs.append(q_out)
        prev = q_out

    return torch.stack(outputs, dim=1)
```

输出：

```python
[B, T, Nq, C]
```

---

## 10. Backward propagation 实现

```python
def _run_backward(self, patch_tokens, q_lrn):
    """
    patch_tokens: [B, T, N, C]
    q_lrn: [B, Nq, C]
    """
    B, T, N, C = patch_tokens.shape
    outputs = [None] * T
    next_q = None

    for t in reversed(range(T)):
        if next_q is None:
            q_in = q_lrn
        else:
            q_in = self.backward_linear(next_q) + q_lrn

        q_out = self.backward_block(q_in, patch_tokens[:, t])
        outputs[t] = q_out
        next_q = q_out

    return torch.stack(outputs, dim=1)
```

输出：

```python
[B, T, Nq, C]
```

---

## 11. 主模型 B23VideoMTWindowModel

文件：

```text
src/models/b23_videomt_window_model.py
```

### 11.1 保留模块

从旧模型中尽量复用：

```text
DINOv3B23Encoder
LoRA 设置
StaticLowResFusion / static adapter
LiteBoundaryDecoder
forensic branch 可选
```

### 11.2 删除模块

新模型中不要实例化：

```text
CCMLite
FGMLite
LowResTFCUFusion that requires f_cc and f_ip
ForgeryCueBank
```

不要出现：

```python
self.ccm = ...
self.fgm = ...
fgm_bank = ...
return_fgm_bank = ...
```

### 11.3 主模型结构

```python
class B23VideoMTWindowModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = DINOv3B23Encoder(...)
        self.query_fusion = WindowQueryFusion(...)
        self.static_adapter = StaticLowResFusion(...)
        self.decoder = LiteBoundaryDecoder(...)
```

如果旧的 `StaticLowResFusion` 强依赖 CCM/FGM 输入，则不要复用旧 fusion，改成更简单的 adapter：

```python
self.feature_proj = nn.Sequential(
    nn.Conv2d(1024, 128, kernel_size=1),
    nn.GroupNorm(8, 128),
    nn.GELU(),
)
```

并把 decoder 输入改为 128 channels。  
原则是：新模型主路径不能再需要 `f_cc` 和 `f_ip`。

---

## 12. 主模型 forward 伪代码

```python
def forward(self, video, mode=None, ablation=None):
    if video.ndim == 5:
        video = video[:, None]

    B, num_windows, num_frames, C, H, W_img = video.shape

    frames = video.reshape(B * num_windows * num_frames, C, H, W_img)

    b23 = self.encoder(frames)
    # [B*num_windows*num_frames, 1024, 32, 32]

    _, feat_c, feat_h, feat_w = b23.shape

    features = b23.reshape(
        B,
        num_windows,
        num_frames,
        feat_c,
        feat_h,
        feat_w,
    )

    logits_per_window = []
    query_states_per_window = []
    edge_per_window = []

    for win_idx in range(num_windows):
        x_win = features[:, win_idx]
        # [B, T, C, Hf, Wf]

        enhanced_win, query_aux = self.query_fusion(x_win)
        # [B, T, C, Hf, Wf]

        enhanced_flat = enhanced_win.reshape(
            B * num_frames,
            feat_c,
            feat_h,
            feat_w,
        )

        f = self.feature_proj(enhanced_flat)
        # [B*T, decoder_channels, Hf, Wf]

        dec_out = self.decoder(f)

        logits = dec_out["logits"]
        # expected [B*T, 1, H, W]

        logits = logits.reshape(B, num_frames, 1, H, W_img)

        logits_per_window.append(logits)
        query_states_per_window.append(query_aux["query_states"])

        edge_logits = dec_out.get("edge_logits", None)
        if edge_logits is None:
            edge_logits = dec_out.get("boundary128", None)

        if edge_logits is not None:
            edge_per_window.append(
                edge_logits.reshape(
                    B,
                    num_frames,
                    1,
                    edge_logits.shape[-2],
                    edge_logits.shape[-1],
                )
            )

    logits = torch.stack(logits_per_window, dim=1)
    # [B, W, T, 1, H, W]

    query_states = torch.stack(query_states_per_window, dim=1)
    # [B, W, T, Nq, C]

    aux = {
        "videomt_queries": query_states,
        "ccm_mask32": None,
        "fgm_mask32": None,
        "fgm_cue": None,
    }

    if len(edge_per_window) > 0:
        aux["edge_logits"] = torch.stack(edge_per_window, dim=1)

    return {
        "logits": logits,
        "aux": aux,
    }
```

---

## 13. Loss：只实现 BCE + Dice，可选 Edge

新增文件：

```text
src/losses/videomt_loss.py
```

默认 loss：

```text
L = BCEWithLogits + Dice
```

可选：

```text
+ 0.2 * EdgeBCE
```

第一版配置中建议不开 edge：

```yaml
loss:
  name: VideoMTLoss
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.0
```

不要添加：

```text
query consistency loss
temporal smoothness loss
ccm loss
fgm loss
cue loss
contrastive loss
flow loss
```

---

## 14. VideoMTLoss 推荐实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def mask_to_edge(mask, kernel_size=5):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class VideoMTLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        edge_weight=0.0,
        edge_kernel_size=5,
        use_pos_weight=False,
        pos_weight=1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_value = float(pos_weight)

    def forward(self, outputs, targets):
        logits = outputs["logits"]

        if isinstance(targets, dict):
            masks = targets.get("masks", targets.get("mask"))
        else:
            masks = targets

        if masks is None:
            raise ValueError("VideoMTLoss requires target masks.")

        if masks.ndim == 5:
            masks = masks.unsqueeze(3)

        masks = masks.float().to(device=logits.device)

        if masks.shape[-2:] != logits.shape[-2:]:
            old_shape = masks.shape
            masks_2d = masks.reshape(-1, 1, old_shape[-2], old_shape[-1])
            masks_2d = F.interpolate(
                masks_2d,
                size=logits.shape[-2:],
                mode="nearest",
            )
            masks = masks_2d.reshape(*logits.shape)

        logits_2d = logits.reshape(-1, 1, logits.shape[-2], logits.shape[-1])
        masks_2d = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1])

        if self.use_pos_weight:
            pos_weight = torch.tensor(
                [self.pos_weight_value],
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            pos_weight = None

        loss_bce = F.binary_cross_entropy_with_logits(
            logits_2d,
            masks_2d,
            pos_weight=pos_weight,
        )

        loss_dice = dice_loss_from_logits(logits_2d, masks_2d)

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        loss_dict = {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_edge": logits_2d.new_tensor(0.0),
        }

        aux = outputs.get("aux", {})
        edge_logits = None

        if isinstance(aux, dict):
            edge_logits = aux.get("edge_logits", None)
            if edge_logits is None:
                edge_logits = aux.get("boundary128", None)

        if self.edge_weight > 0 and edge_logits is not None:
            edge_logits_2d = edge_logits.reshape(
                -1,
                1,
                edge_logits.shape[-2],
                edge_logits.shape[-1],
            )

            edge_target = F.interpolate(
                masks_2d,
                size=edge_logits_2d.shape[-2:],
                mode="nearest",
            )
            edge_target = mask_to_edge(
                edge_target,
                kernel_size=self.edge_kernel_size,
            )

            loss_edge = F.binary_cross_entropy_with_logits(
                edge_logits_2d,
                edge_target,
            )

            total = total + self.edge_weight * loss_edge
            loss_dict["loss_edge"] = loss_edge

        loss_dict["loss_total"] = total
        return total, loss_dict
```

---

## 15. 配置文件

新增：

```text
configs/b23_videomt_window.yaml
```

建议内容：

```yaml
model:
  name: B23VideoMTWindowModel

input_size: 512
num_clips: 4
num_frames: 4
clip_stride: 1

dinov3:
  repo: ./dinov3
  weights: ./weights/dinov3.pth
  model_name: dinov3_vitl16
  input_size: 512
  patch_size: 16
  output_block: 23
  output_resolution: 32
  feature_dim: 1024
  freeze_backbone: true

lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.1
  targets: qkv,proj,fc1,fc2
  layers: all

videomt:
  enabled: true
  dim: 1024
  num_queries: 16
  heads: 8
  ffn_ratio: 4.0
  dropout: 0.0
  bidirectional: true
  residual_alpha_init: 0.0

tfcu:
  version: videomt_window
  ccm:
    enabled: false
  fgm:
    enabled: false

decoder:
  out_channels: 128

loss:
  name: VideoMTLoss
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.0
  edge_kernel_size: 5
  use_pos_weight: false
  pos_weight: 1.0
```

---

## 16. 修改 src/models/__init__.py

加入：

```python
from .b23_videomt_window_model import B23VideoMTWindowModel
```

并更新：

```python
__all__ = [
    "B23TFCUCCMFGMLiteModel",
    "B23VideoMTWindowModel",
]
```

---

## 17. 修改模型构建逻辑

如果当前 trainer 直接写死：

```python
model = B23TFCUCCMFGMLiteModel(cfg)
```

需要改成：

```python
from src.models import B23TFCUCCMFGMLiteModel, B23VideoMTWindowModel


def build_model(cfg):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "B23TFCUCCMFGMLiteModel")

    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)

    if name == "B23TFCUCCMFGMLiteModel":
        return B23TFCUCCMFGMLiteModel(cfg)

    raise ValueError(f"Unknown model name: {name}")
```

然后：

```python
model = build_model(cfg)
```

---

## 18. 修改 loss 构建逻辑

新增：

```python
from src.losses.videomt_loss import VideoMTLoss
```

实现：

```python
def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "VideoMTLoss")

    if name == "VideoMTLoss":
        return VideoMTLoss(
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            edge_weight=loss_cfg.get("edge_weight", 0.0),
            edge_kernel_size=loss_cfg.get("edge_kernel_size", 5),
            use_pos_weight=loss_cfg.get("use_pos_weight", False),
            pos_weight=loss_cfg.get("pos_weight", 1.0),
        )

    raise ValueError(f"Unknown loss: {name}")
```

训练时：

```python
outputs = model(images)
loss, loss_dict = criterion(outputs, masks)
```

日志只记录：

```text
loss_total
loss_bce
loss_dice
loss_edge
```

---

## 19. 去掉新模型训练中的 FGM bank 依赖

旧训练代码可能有：

```python
fgm_bank = model.new_fgm_bank(...)
outputs = model(..., fgm_bank=fgm_bank, return_fgm_bank=True)
```

新模型不支持这些参数。

为了兼容旧模型和新模型，写成：

```python
inner = model.module if hasattr(model, "module") else model
supports_fgm_bank = hasattr(inner, "new_fgm_bank")

if supports_fgm_bank:
    outputs = model(
        images,
        mode=mode,
        ablation=ablation,
        fgm_bank=fgm_bank,
        return_fgm_bank=True,
    )
else:
    outputs = model(
        images,
        mode=mode,
        ablation=ablation,
    )
```

新模型 forward 签名不要出现：

```python
fgm_bank
return_fgm_bank
```

---

## 20. 最小 Smoke Test

新增或临时运行：

```python
import torch

from src.models import B23VideoMTWindowModel
from src.utils.config import load_config

cfg = load_config("configs/b23_videomt_window.yaml")

model = B23VideoMTWindowModel(cfg).cuda().eval()

x = torch.randn(1, 2, 4, 3, 512, 512).cuda()

with torch.no_grad():
    out = model(x)

assert "logits" in out
assert "aux" in out
assert out["logits"].shape == (1, 2, 4, 1, 512, 512)
assert "videomt_queries" in out["aux"]
assert out["aux"]["videomt_queries"].shape[:3] == (1, 2, 4)
```

Loss 测试：

```python
from src.losses.videomt_loss import VideoMTLoss

criterion = VideoMTLoss(bce_weight=1.0, dice_weight=1.0, edge_weight=0.0).cuda()

mask = torch.randint(0, 2, (1, 2, 4, 1, 512, 512)).float().cuda()

loss, loss_dict = criterion(out, mask)

assert torch.isfinite(loss)
assert "loss_bce" in loss_dict
assert "loss_dice" in loss_dict
```

---

## 21. 训练优先级

当前只需要跑最终版：

```text
B23VideoMTWindowModel
+ bidirectional=True
+ BCE + Dice
```

不需要先跑：

```text
A1 static baseline
A2 learned query only
A3 propagation only
A4 forward only
```

这些后续做论文实验时再补。

当前第一目标：

```text
1. 模型能 forward。
2. loss 能 backward。
3. 训练脚本能跑起来。
4. 不依赖 CCM/FGM/FGM bank。
5. 显存可接受。
6. loss 能下降。
```

---

## 22. 后续可选实验，不在本次实现范围

最终版稳定后，再补以下实验：

```text
A0: 原 v1 DINOv3 + CCM + FGM
A1: DINOv3 + decoder，禁用 CCM/FGM
A2: learned queries only
A3: query propagation only
A4: one-way VidEoMT-style query fusion
A5: bidirectional VidEoMT-style query fusion
```

Loss 消融后续再做：

```text
C1: BCE
C2: BCE + Dice
C3: BCE + Dice + Edge
```

这些不是本次 agent 的首要任务。

---

## 23. 最终验收标准

完成后必须满足：

```text
1. B23VideoMTWindowModel 可以正常 import。
2. 输入 [B,W,T,3,H,W] 输出 [B,W,T,1,H,W]。
3. 输入 [B,T,3,H,W] 可以自动转成单 window。
4. 新模型不实例化 CCMLite。
5. 新模型不实例化 FGMLite。
6. 新模型不使用 FGM bank。
7. 新模型没有 gate_f / gate_b / sigmoid MLP gate。
8. Query fusion 使用 Linear(prev_query) + learned_query。
9. Bidirectional fusion 使用 forward/backward average。
10. Loss 默认只有 BCE + Dice。
11. Edge loss 默认关闭。
12. 训练代码可以通过 model.name 选择新旧模型。
13. 旧模型不被破坏。
14. Smoke test 能通过。
15. 一次 mini-batch 训练能完成 forward、loss、backward、optimizer step。
```

---

## 24. 最终一句话总结

本次实现不是“在原 CCM/FGM 上再加一个模块”，而是：

```text
用 VidEoMT-style propagated forgery queries
替换 CCM/FGM 这两个手工时序模块，
并用最简单的 BCE + Dice 端到端训练。
```
