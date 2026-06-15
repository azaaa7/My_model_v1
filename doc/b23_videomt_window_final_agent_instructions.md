# b23_videomt_window 最终版代码修改指令

> 目标：把当前 `b23_videomt_window` 从“DINOv3 B23 特征后处理 + query residual”改成更接近 VidEoMT 思想的最终结构：**query 在 DINOv3 最后 4 个 block 内与 patch token 同步交互，使用 `Linear(prev_q) + Q_lrn` 做在线传播，跨 window 延续 query state，并让 query 直接预测 mask**。  
> 不要再通过 `mean(query) -> Linear -> features + alpha * residual` 的方式注入时间信息；不要添加门控网络；不要引入新的重型 tracker / decoder。

---

## 0. 当前项目中必须处理的关键问题

当前项目的 `b23_videomt_window` 结构大致是：

```text
DINOv3B23Encoder
  -> WindowQueryFusion
      -> QueryPatchBlock
      -> query_states.mean(dim=2)
      -> query_to_feature
      -> features + residual_alpha * query_context
  -> feature_proj
  -> LiteBoundaryDecoder
  -> logits
```

这和最终目标不一致。需要把它改成：

```text
DINOv3QueryEncoder
  frame t:
    前 20 个 DINOv3 blocks 只处理 patch/class token
    最后 4 个 DINOv3 blocks 同时处理 patch token + query token

    t = 0:
      q_in = Q_lrn
    t > 0:
      q_in = Linear(q_{t-1}) + Q_lrn

  -> patch feature for decoder
  -> decoder feature map, especially f128
  -> query mask head:
       mask_embed(q_t) · mask_feature(f128)
       query score
       logsumexp aggregation
  -> final logits
```

核心修改点：

1. **不要再使用 query residual。**
2. **不要再使用 bidirectional query。**
3. **query 要进入 DINOv3 最后 4 个 block。**
4. **query 要直接输出 query-level masks。**
5. **跨 window 要携带上一 window 的最后一帧 query。**
6. **loss 要增加 query-level mask supervision。**
7. **训练入口、loss registry、optimizer param group、test/eval 状态传递都要同步修改。**

---

## 1. 新增最终版配置文件

新增文件：

```text
configs/b23_videomt_window_final.yaml
```

内容如下：

```yaml
type: train
seed: 666666

model:
  name: "B23VideoMTWindowModel"
  version: "videomt_encoder_query_mask_final"
  input_size: 512

batch_size: 2
num_clips: 1
num_frames: 5
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

augment_prob: 0.70
spatial_augment_prob: 0.70
appearance_augment_prob: 0.40

temporal_augment:
  frame_swap:
    enabled: true
    prob: 0.05
    start_epoch: 80
    max_swaps: 1
    local_radius: 2
  frame_drop:
    enabled: true
    prob: 0.05
    start_epoch: 80
    max_drops: 1

amp: true
use_activation_checkpoint: true
grad_accum_steps: 2
num_workers: 4
log_interval: 20

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

  query_injection:
    enabled: true
    mode: "last_blocks"
    inject_after_block: 19
    query_blocks: [20, 21, 22, 23]
    num_query_blocks: 4
    keep_cls_token: true
    patch_query_attention: true

lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.05
  layers: "last4"
  targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"

videomt:
  enabled: true
  dim: 1024
  num_queries: 32
  heads: 16
  ffn_ratio: 4.0
  dropout: 0.0
  bidirectional: false

  residual:
    enabled: false
    residual_alpha_init: 0.0

  propagation:
    enabled: true
    type: "linear_plus_learned"
    first_frame: "learned"
    prev_linear: true
    detach_within_window: false
    stateful_windows: true
    detach_across_windows: true
    carry_state_in_eval: true
    reset_state_per_video: true

  query_fusion:
    enabled: true
    formula: "linear_prev_plus_learned"
    linear_bias: true
    linear_init: "xavier_uniform"

  query_mask_head:
    enabled: true
    mask_dim: 256
    mask_embed_mlp_layers: 3
    mask_feature_source: "decoder_128"
    mask_resolution: 128
    upsample_to_input: true

    score_head:
      enabled: true
      hidden_dim: 256

    aggregation:
      type: "logsumexp"
      temperature: 1.0
      use_query_score: true

tfcu:
  version: "videomt_window_final"
  ccm:
    enabled: false
  fgm:
    enabled: false

decoder:
  type: "lite_boundary"
  input_resolution: 32
  in_channels: 128
  stages:
    - {resolution: 64, channels: 96}
    - {resolution: 128, channels: 48}
    - {resolution: 256, channels: 16}
  final_upsample: "bilinear"
  output_512_multichannel: false

  export_features:
    enabled: true
    resolutions: [128]

  mask128_head:
    enabled: false

  boundary_head:
    enabled: true
    resolution: 128

loss:
  type: "videomt_query_mask"
  name: "VideoMTQueryMaskLoss"

  bce_weight: 1.0
  dice_weight: 1.0

  query_mask:
    enabled: true
    bce_weight: 0.5
    dice_weight: 0.5

    matching:
      type: "hungarian_connected_components"
      max_components: 32
      min_component_area: 16
      keep_assignment_across_frames: true
      new_object_on_first_appearance: true
      iou_threshold: 0.05
      center_distance_threshold: 64

    no_object:
      enabled: true
      weight: 0.1

  edge_weight: 0.05
  edge_kernel_size: 5

  use_pos_weight: true
  pos_weight: 2.0
  dice_eps: 1.0e-6

optimizer:
  type: "adamw"
  learning_rate: 1.0e-4
  lr_lora: 5.0e-5
  lr_decoder: 1.0e-4
  lr_query: 1.0e-4
  lr_mask_head: 1.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.999]
  eps: 1.0e-8

  layer_wise_lr_decay:
    enabled: true
    decay_rate: 0.6

scheduler:
  type: "poly"
  warmup_epochs: 10
  power: 0.9
  min_lr: 1.0e-6

val_full_video: true
val_test_max_clips: 9999
val_num_workers: 0

test_full_video: true
test_max_clips: 9999

train:
  n_epochs: 1000
  save_dir: "runs/b23_videomt_window_final"
  val_interval: 10
  max_grad_norm: 1.0
  skip_nonfinite: true

  ema:
    enabled: true
    decay: 0.999

  save_best_by: "val_iou"

stability:
  logit_clamp: 30.0
  nan_to_num: true

ddp:
  auto_torchrun: true
  cuda_visible_devices: "4,5"
  nproc_per_node: 2
  dist_backend: "nccl"
  find_unused_parameters: false
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  torchrun_log_dir: "runs/b23_videomt_window_final/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
  log_query_stats: true
  log_query_assignment: false
```

---

## 2. 文件级修改清单

必须修改或新增这些文件：

```text
configs/b23_videomt_window_final.yaml

src/models/__init__.py
src/models/b23_videomt_window_model.py
src/models/dinov3_b23_encoder.py
src/models/videomt/__init__.py
src/models/videomt/query_encoder.py              # 新增，推荐
src/models/videomt/query_mask_head.py            # 新增
src/models/decoders/lite_boundary_decoder.py

src/losses/__init__.py
src/losses/videomt_query_mask_loss.py            # 新增

src/train/trainer.py
src/train/optimizer.py
src/train/scheduler.py
src/data/...                                     # 只在 temporal augment 不支持 start_epoch 时修改
src/eval/tester.py                               # 如 test 仍写死旧模型，也要改
```

不建议继续维护旧的 `WindowQueryFusion` 作为最终路径。可以保留文件用于历史对比，但最终模型不要再调用它。

---

## 3. 修改 `src/models/__init__.py`

当前 `trainer.py` 可能只从 `src.models` 导入旧模型。必须导出新模型：

```python
from .b23_tfcu_ccm_fgm_model import B23TFCUCCMFGMLiteModel
from .b23_videomt_window_model import B23VideoMTWindowModel
from .dinov3_b23_encoder import DINOv3B23Encoder, load_dinov3_backbone

__all__ = [
    "B23TFCUCCMFGMLiteModel",
    "B23VideoMTWindowModel",
    "DINOv3B23Encoder",
    "load_dinov3_backbone",
]
```

验收标准：

```bash
python - <<'PY'
from src.models import B23VideoMTWindowModel
print(B23VideoMTWindowModel)
PY
```

---

## 4. 改造 DINOv3 encoder：支持 query token 进入最后 4 个 block

### 4.1 目标接口

在 `src/models/dinov3_b23_encoder.py` 中保留原 `DINOv3B23Encoder.forward(frames)`，同时新增一个方法：

```python
def forward_with_queries(
    self,
    frames: torch.Tensor,
    query_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        frames: [N, 3, H, W]
        query_tokens: [N, Q, C]

    Returns:
        patch_features: [N, C, 32, 32]
        out_queries: [N, Q, C]
    """
```

### 4.2 实现要求

不要使用 `get_intermediate_layers(...)` 来做 query 注入，因为这个 API 只返回 patch 中间特征，不会接受 query token。需要直接调用 DINOv3 ViT 的内部 patch embed / blocks / norm。

Agent 必须先在本地检查 backbone 的真实属性名：

```python
print(type(self.backbone))
print([name for name, _ in self.backbone.named_children()])
```

通常 ViT 会包含类似：

```text
patch_embed
cls_token
pos_embed
blocks
norm
```

但不能硬猜，必须兼容实际 DINOv3 实现。

### 4.3 推荐实现骨架

在 `DINOv3B23Encoder.__init__` 中读取：

```python
qcfg = dinov3_cfg.get("query_injection", {}) or {}
self.query_injection_enabled = bool(qcfg.get("enabled", False))
self.inject_after_block = int(qcfg.get("inject_after_block", 19))
self.query_blocks = list(qcfg.get("query_blocks", [20, 21, 22, 23]))
self.keep_cls_token = bool(qcfg.get("keep_cls_token", True))
```

新增 helper：

```python
def _patchify_tokens(self, frames: torch.Tensor) -> torch.Tensor:
    """
    Return backbone token sequence before transformer blocks.
    Must include cls/register tokens exactly as the native backbone expects.
    """
    # 必须根据本地 DINOv3 实现适配：
    # 1. normalize frames
    # 2. patch_embed
    # 3. add cls token / register tokens / pos embed
    # 4. return tokens [N, 1+P(+R), C]
```

新增主逻辑：

```python
def forward_with_queries(self, frames: torch.Tensor, query_tokens: torch.Tensor):
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N,3,H,W], got {tuple(frames.shape)}")
    if query_tokens.ndim != 3:
        raise ValueError(f"query_tokens must be [N,Q,C], got {tuple(query_tokens.shape)}")

    frames = (frames - self.image_mean) / self.image_std
    grad_enabled = torch.is_grad_enabled() and (not self.freeze_backbone or self.use_lora)

    with torch.set_grad_enabled(grad_enabled):
        tokens = self._patchify_tokens_aligned_with_dinov3(frames)

        # blocks 0..19: only native tokens
        for i, blk in enumerate(self.backbone.blocks):
            if i <= self.inject_after_block:
                tokens = blk(tokens)
            else:
                break

        # insert query tokens after native tokens
        native_len = tokens.shape[1]
        tokens = torch.cat([tokens, query_tokens], dim=1)

        # blocks 20..23: native tokens + query tokens together
        for i in range(self.inject_after_block + 1, self.output_block + 1):
            tokens = self.backbone.blocks[i](tokens)

        tokens = self.backbone.norm(tokens)

        native_tokens = tokens[:, :native_len]
        out_queries = tokens[:, native_len:]

        # remove cls/register tokens, keep patch tokens only
        patch_tokens = self._extract_patch_tokens(native_tokens)

        h = w = self.output_resolution
        patch_features = patch_tokens.transpose(1, 2).reshape(frames.shape[0], -1, h, w)

    return patch_features, out_queries
```

### 4.4 `_extract_patch_tokens` 的要求

必须正确处理：

- cls token
- register tokens，如果 DINOv3 有
- patch tokens 数量必须等于 `32 * 32 = 1024`

推荐写法：

```python
def _extract_patch_tokens(self, native_tokens: torch.Tensor) -> torch.Tensor:
    h = w = self.output_resolution
    num_patches = h * w

    if native_tokens.shape[1] < num_patches:
        raise RuntimeError(...)

    # 最稳妥：从末尾取 num_patches 个 token
    # 因为 cls/register token 通常在 patch token 前面。
    patch_tokens = native_tokens[:, -num_patches:, :]

    if patch_tokens.shape[1] != num_patches:
        raise RuntimeError(...)

    return patch_tokens
```

验收标准：

```python
enc = DINOv3B23Encoder(dinov3_cfg, lora_cfg).cuda()
x = torch.rand(2, 3, 512, 512, device="cuda")
q = torch.randn(2, 32, 1024, device="cuda")
f, qo = enc.forward_with_queries(x, q)
assert f.shape == (2, 1024, 32, 32)
assert qo.shape == (2, 32, 1024)
```

---

## 5. 新增 `src/models/videomt/query_mask_head.py`

新增 query mask head，让每个 query 直接预测一个 mask。不要把 query 平均后做 residual。

### 5.1 文件内容骨架

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    for i in range(num_layers):
        src = in_dim if i == 0 else hidden_dim
        dst = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(src, dst))
        if i != num_layers - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class QueryMaskHead(nn.Module):
    """
    Convert query tokens into query-level masks and aggregate them into final binary logits.

    Inputs:
        query_states: [B, W, T, Q, C] or [B, T, Q, C]
        mask_features: [B, W, T, Cmask, Hm, Wm] or [B, T, Cmask, Hm, Wm]

    Outputs:
        {
          "logits": [B, W, T, 1, Hout, Wout],
          "query_logits": [B, W, T, Q, Hm, Wm],
          "query_scores": [B, W, T, Q, 1],
        }
    """

    def __init__(self, query_dim: int, in_mask_channels: int, cfg: dict[str, Any]):
        super().__init__()
        self.mask_dim = int(cfg.get("mask_dim", 256))
        self.mask_resolution = int(cfg.get("mask_resolution", 128))
        self.upsample_to_input = bool(cfg.get("upsample_to_input", True))

        num_layers = int(cfg.get("mask_embed_mlp_layers", 3))
        self.mask_embed = build_mlp(query_dim, query_dim, self.mask_dim, num_layers)
        self.mask_feature_proj = nn.Conv2d(in_mask_channels, self.mask_dim, kernel_size=1)

        score_cfg = cfg.get("score_head", {}) or {}
        score_hidden = int(score_cfg.get("hidden_dim", 256))
        self.use_score = bool(score_cfg.get("enabled", True))
        self.score_head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, 1),
        )

        agg_cfg = cfg.get("aggregation", {}) or {}
        self.aggregation = str(agg_cfg.get("type", "logsumexp"))
        self.temperature = float(agg_cfg.get("temperature", 1.0))
        self.use_query_score = bool(agg_cfg.get("use_query_score", True))

    def forward(
        self,
        query_states: torch.Tensor,
        mask_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        squeeze_window = False
        if query_states.ndim == 4:
            query_states = query_states[:, None]
            mask_features = mask_features[:, None]
            squeeze_window = True

        b, w, t, q, c = query_states.shape
        _, _, _, cm, hm, wm = mask_features.shape

        q_flat = query_states.reshape(b * w * t, q, c)
        f_flat = mask_features.reshape(b * w * t, cm, hm, wm)

        mask_embed = self.mask_embed(q_flat)                       # [BWT,Q,Dm]
        mask_feat = self.mask_feature_proj(f_flat)                 # [BWT,Dm,Hm,Wm]
        query_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_feat)

        query_scores = self.score_head(q_flat) if self.use_score else torch.zeros(
            q_flat.shape[0], q, 1, device=q_flat.device, dtype=q_flat.dtype
        )

        if self.use_query_score:
            query_logits_for_agg = query_logits + query_scores[..., None]
        else:
            query_logits_for_agg = query_logits

        if self.aggregation == "logsumexp":
            temp = max(self.temperature, 1.0e-6)
            logits = torch.logsumexp(query_logits_for_agg / temp, dim=1, keepdim=True) * temp
        elif self.aggregation == "max":
            logits = query_logits_for_agg.max(dim=1, keepdim=True).values
        elif self.aggregation == "mean":
            logits = query_logits_for_agg.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown query mask aggregation: {self.aggregation}")

        if self.upsample_to_input and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        query_logits = query_logits.reshape(b, w, t, q, hm, wm)
        query_scores = query_scores.reshape(b, w, t, q, 1)
        logits = logits.reshape(b, w, t, 1, logits.shape[-2], logits.shape[-1])

        if squeeze_window:
            logits = logits[:, 0]
            query_logits = query_logits[:, 0]
            query_scores = query_scores[:, 0]

        return {
            "logits": logits,
            "query_logits": query_logits,
            "query_scores": query_scores,
        }
```

---

## 6. 修改 decoder：导出 f128 特征

修改 `src/models/decoders/lite_boundary_decoder.py`，让 `forward()` 返回 `features128`。

当前返回中加入：

```python
return {
    "mask128": mask128,
    "boundary128": boundary128,
    "mask256": mask256,
    "logits": logits,
    "features128": f128,
    "debug": debug,
}
```

验收标准：

```python
dec_out = self.decoder(dec_in)
assert "features128" in dec_out
assert dec_out["features128"].shape[-2:] == (128, 128)
```

---

## 7. 新增 VidEoMT query encoder wrapper

新增文件：

```text
src/models/videomt/query_encoder.py
```

作用：管理 learned queries、`Linear(prev_q)+Q_lrn`、window 内传播、跨 window state。

### 7.1 文件骨架

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VideoMTQueryController(nn.Module):
    """
    Maintains learned queries and converts previous-frame queries into current-frame inputs.

    Formula:
        t = 0: q_in = Q_lrn
        t > 0: q_in = Linear(q_prev) + Q_lrn
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.dim = int(cfg.get("dim", 1024))
        self.num_queries = int(cfg.get("num_queries", 32))

        prop_cfg = cfg.get("propagation", {}) or {}
        self.detach_within_window = bool(prop_cfg.get("detach_within_window", False))
        self.detach_across_windows = bool(prop_cfg.get("detach_across_windows", True))

        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.dim) * 0.02)
        self.prev_linear = nn.Linear(self.dim, self.dim, bias=bool((cfg.get("query_fusion", {}) or {}).get("linear_bias", True)))

        linear_init = str((cfg.get("query_fusion", {}) or {}).get("linear_init", "xavier_uniform"))
        if linear_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.prev_linear.weight)
            if self.prev_linear.bias is not None:
                nn.init.zeros_(self.prev_linear.bias)

    def initial_queries(self, batch_size: int, device, dtype) -> torch.Tensor:
        return self.learned_queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def make_input_queries(
        self,
        batch_size: int,
        device,
        dtype,
        prev_q: torch.Tensor | None,
        detach_prev: bool = False,
    ) -> torch.Tensor:
        q_lrn = self.initial_queries(batch_size, device, dtype)

        if prev_q is None:
            return q_lrn

        if detach_prev:
            prev_q = prev_q.detach()

        return self.prev_linear(prev_q.to(dtype=dtype)) + q_lrn
```

---

## 8. 改造 `src/models/b23_videomt_window_model.py`

这是最关键的文件。最终模型不要再调用 `WindowQueryFusion`。

### 8.1 imports

替换：

```python
from .videomt import WindowQueryFusion
```

为：

```python
from .videomt.query_encoder import VideoMTQueryController
from .videomt.query_mask_head import QueryMaskHead
```

### 8.2 `__init__` 修改

删除：

```python
self.query_fusion = WindowQueryFusion(videomt_cfg)
```

新增：

```python
self.query_controller = VideoMTQueryController(videomt_cfg)

qmh_cfg = videomt_cfg.get("query_mask_head", {}) or {}
mask_head_in_channels = int(decoder_cfg["stages"][1].get("channels", 48))  # f128 channels
self.query_mask_head = QueryMaskHead(
    query_dim=self.feature_dim,
    in_mask_channels=mask_head_in_channels,
    cfg=qmh_cfg,
)
```

保留：

```python
self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
self.feature_proj = ...
self.decoder = LiteBoundaryDecoder(decoder_cfg)
```

### 8.3 新增 `encode_frame_with_query`

```python
def encode_frame_with_query(self, frame: torch.Tensor, q_in: torch.Tensor):
    if self.use_activation_checkpoint and self.training:
        return checkpoint(self.encoder.forward_with_queries, frame, q_in, use_reentrant=False)
    return self.encoder.forward_with_queries(frame, q_in)
```

### 8.4 重写 `forward`

最终逻辑：

```python
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
        raise ValueError("Final VidEoMT model does not support disable_videomt.")

    if video.ndim == 5:
        video = video[:, None]
    if video.ndim != 6:
        raise ValueError(f"video must be [B,W,T,3,H,W], got {tuple(video.shape)}")

    batch, num_windows, num_frames, channels, height, width = video.shape
    device = video.device

    prev_q = None
    if isinstance(videomt_state, dict):
        prev_q = videomt_state.get("prev_q")

    logits_per_window = []
    query_states_per_window = []
    query_logits_per_window = []
    query_scores_per_window = []
    edge_per_window = []
    debug = {
        "input_video_shape": tuple(video.shape),
        "videomt_final": True,
        "query_injection": True,
    }

    prop_cfg = ((self.cfg.get("videomt", {}) or {}).get("propagation", {}) or {})
    detach_across_windows = bool(prop_cfg.get("detach_across_windows", True))
    detach_within_window = bool(prop_cfg.get("detach_within_window", False))

    for win_idx in range(num_windows):
        patch_features_this_window = []
        queries_this_window = []

        for frame_idx in range(num_frames):
            frame = video[:, win_idx, frame_idx]

            detach_prev = bool(
                prev_q is not None
                and win_idx > 0
                and frame_idx == 0
                and detach_across_windows
            )

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

        patch_features = torch.stack(patch_features_this_window, dim=1)  # [B,T,C,32,32]
        query_states = torch.stack(queries_this_window, dim=1)           # [B,T,Q,C]

        feat_flat = patch_features.reshape(batch * num_frames, self.feature_dim, patch_features.shape[-2], patch_features.shape[-1])
        dec_in = self.feature_proj(feat_flat)
        dec_out = self.decoder(dec_in)

        f128 = dec_out["features128"].reshape(
            batch, num_frames,
            dec_out["features128"].shape[1],
            dec_out["features128"].shape[-2],
            dec_out["features128"].shape[-1],
        )

        qmh_out = self.query_mask_head(
            query_states=query_states,
            mask_features=f128,
            output_size=(height, width),
        )

        logits = qmh_out["logits"]  # [B,T,1,H,W]
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
```

注意：

- `return_videomt_state` 用于 eval/test full video 跨 batch/window 传递。
- 模型必须接受 `**kwargs`，因为当前 trainer 可能仍会传 `fgm_bank`、`return_fgm_bank`。最终模型可以忽略这些参数，但不能报错。
- 如果训练入口仍传 `fgm_bank`，不要在最终模型里调用 `new_fgm_bank`。

---

## 9. 修改 `src/models/videomt/__init__.py`

```python
from .query_encoder import VideoMTQueryController
from .query_mask_head import QueryMaskHead

__all__ = [
    "VideoMTQueryController",
    "QueryMaskHead",
]
```

可以保留旧 `WindowQueryFusion`，但不要在最终模型默认路径中导出为主模块。

---

## 10. 新增 query-level loss

新增：

```text
src/losses/videomt_query_mask_loss.py
```

该 loss 包含：

1. 主输出 `logits` 的 BCE + Dice。
2. `aux["query_logits"]` 的 query-level BCE + Dice。
3. 可选 edge loss。
4. 伪实例匹配：从二值 GT mask 中提 connected components，分配给 query。

### 10.1 简化且稳定的第一版实现原则

不要一开始写复杂跨帧 Hungarian 跟踪。第一版以稳定为主：

- 对每个 frame 的 GT 二值 mask 做 connected components。
- 每帧最多保留 `max_components = num_queries` 个组件。
- 用 query mask 和组件 mask 的 Dice cost 做 Hungarian 匹配。
- 匹配到的 query 用对应组件监督。
- 没匹配到的 query 用全 0 mask，以 `no_object.weight` 轻量监督。
- 后续可以再把 `keep_assignment_across_frames` 做成真正跨帧一致。

这虽然不是最完整的视频实例监督，但已经比当前只监督最终 logits 强很多，并且不会引入新的 tracker/gate。

### 10.2 connected components 可以先用 CPU fallback

由于 GT 分辨率 512，query logits 是 128，可以把 GT resize 到 query mask 分辨率后再做 components。为了减少依赖，可以先用 `scipy.ndimage.label`；如果环境没有 scipy，再 fallback 到 OpenCV 或纯 PyTorch flood fill。

推荐先尝试：

```python
try:
    from scipy import ndimage
except Exception:
    ndimage = None
```

### 10.3 Loss 类接口

```python
class VideoMTQueryMaskLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        ...

    def forward(
        self,
        outputs,
        targets: torch.Tensor,
        aux: dict[str, Any] | None = None,
        epoch: int | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...
```

必须兼容当前 trainer 的调用方式：

```python
criterion(logits, target, aux=aux, epoch=epoch, include_aux=True)
```

和：

```python
criterion(outputs, targets, aux=aux, epoch=epoch, include_aux=True)
```

建议让 `_extract_logits_and_aux` 同时支持 dict 和 tensor。

### 10.4 关键张量形状

主 logits：

```text
logits: [B,W,T,1,512,512] 或 [B,T,1,512,512]
targets: [B,W,T,1,512,512] / [B,T,1,512,512] / [B,1,512,512]
```

query logits：

```text
aux["query_logits"]: [B,W,T,Q,128,128]
aux["query_scores"]: [B,W,T,Q,1]
```

query target：

```text
query_targets: [B,W,T,Q,128,128]
query_valid:   [B,W,T,Q]
```

### 10.5 Loss item 必须返回这些 key

```python
items = {
    "loss_total": ...,
    "loss_bce": ...,
    "loss_dice": ...,
    "loss_query_bce": ...,
    "loss_query_dice": ...,
    "loss_query_no_object": ...,
    "loss_edge": ...,
    "main_loss": ...,
}
```

这样训练日志能看到 query loss 是否有效。

---

## 11. 修改 `src/losses/__init__.py`

加入：

```python
from .videomt_query_mask_loss import VideoMTQueryMaskLoss
```

并加入 `__all__`：

```python
"VideoMTQueryMaskLoss",
```

---

## 12. 修改 `src/train/trainer.py`

当前 trainer 可能写死：

```python
from src.models import B23TFCUCCMFGMLiteModel
...
model = B23TFCUCCMFGMLiteModel(cfg).to(device)
```

必须改为按配置创建模型。

### 12.1 新增 model builder

```python
from src.models import B23TFCUCCMFGMLiteModel, B23VideoMTWindowModel
from src.losses import (
    AuxiliaryLoss,
    CompositeForensicLoss,
    SegmentationLoss,
    SUMILocalizationLoss,
    TTFMinimalLoss,
    VideoMTLoss,
    VideoMTQueryMaskLoss,
)

def build_model(cfg: dict[str, Any]):
    name = str((cfg.get("model", {}) or {}).get("name", "B23TFCUCCMFGMLiteModel"))
    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)
    if name == "B23TFCUCCMFGMLiteModel":
        return B23TFCUCCMFGMLiteModel(cfg)
    raise ValueError(f"Unknown model.name: {name}")
```

然后替换：

```python
model = B23TFCUCCMFGMLiteModel(cfg).to(device)
```

为：

```python
model = build_model(cfg).to(device)
```

### 12.2 修改 loss builder

```python
def build_loss(cfg: dict[str, Any]):
    loss_cfg = cfg.get("loss", {}) or {}
    loss_type = str(loss_cfg.get("type", "")).lower()

    if loss_type == "videomt_query_mask":
        return VideoMTQueryMaskLoss(loss_cfg), None

    if loss_type == "videomt":
        return VideoMTLoss(loss_cfg), None

    if loss_type == "composite_forensic":
        return CompositeForensicLoss(loss_cfg), None

    if loss_type == "ttf_minimal":
        return TTFMinimalLoss(loss_cfg), None

    return SegmentationLoss(loss_cfg), AuxiliaryLoss(cfg.get("aux_loss", {}))
```

### 12.3 移除/绕过旧 FGM bank 对最终模型的强依赖

当前 trainer 可能调用：

```python
fgm_bank = _new_fgm_bank(model)
out = model(images, mode="train", fgm_bank=fgm_bank, return_fgm_bank=use_stateful_bank, epoch=epoch)
```

最终 VidEoMT 不需要 FGM bank。要新增 VidEoMT state 管理：

```python
def _is_videomt_model(cfg: dict[str, Any]) -> bool:
    return str((cfg.get("model", {}) or {}).get("name", "")) == "B23VideoMTWindowModel"

def _videomt_stateful_enabled(cfg: dict[str, Any], mode: str) -> bool:
    if not _is_videomt_model(cfg):
        return False
    prop = ((cfg.get("videomt", {}) or {}).get("propagation", {}) or {})
    if mode == "train":
        return bool(prop.get("stateful_windows", False))
    return bool(prop.get("carry_state_in_eval", True))
```

训练 loop 中：

```python
use_videomt_stateful = _videomt_stateful_enabled(cfg, "train")
videomt_state = None
current_video_id = None
...
if use_videomt_stateful and isinstance(batch, dict):
    video_id = _batch_str(batch, "video_id", _name)
    should_reset = videomt_state is None or video_id != current_video_id or _batch_bool(batch, "is_first_window", False)
    if should_reset:
        videomt_state = None
        current_video_id = video_id
```

调用模型：

```python
if _is_videomt_model(cfg):
    out = model(
        images,
        mode="train",
        epoch=epoch,
        videomt_state=videomt_state,
        return_videomt_state=use_videomt_stateful,
    )
    if use_videomt_stateful:
        videomt_state = out.get("videomt_state", videomt_state)
else:
    out = model(images, mode="train", fgm_bank=fgm_bank, return_fgm_bank=use_stateful_bank, epoch=epoch)
```

eval 中同理：

```python
out = model(
    images,
    mode="eval",
    ablation=ablation,
    videomt_state=videomt_state,
    return_videomt_state=use_videomt_stateful,
)
```

### 12.4 对齐 logits/masks 时保留 aux query logits

当前 `_filter_by_valid_mask` 可能只处理 `aux` 中与 valid mask shape 匹配的张量。确保它不会错误 reshape `query_logits`。建议加入对 query logits 的处理：

```python
if key in ("query_logits", "query_scores", "videomt_queries"):
    # query_logits starts with [B,W,T,...], so if valid_mask is [B,W,T], flatten first dims.
    ...
```

如果 valid mask 暂时没有用于该配置，可以先跳过过滤 query aux，但不要让它报错。

---

## 13. 修改 optimizer param group

文件：

```text
src/train/optimizer.py
```

必须支持：

```yaml
optimizer:
  lr_lora: 5.0e-5
  lr_decoder: 1.0e-4
  lr_query: 1.0e-4
  lr_mask_head: 1.0e-4
```

推荐分组逻辑：

```python
def build_optimizer(model, cfg):
    opt_cfg = cfg.get("optimizer", {}) or {}
    base_lr = float(opt_cfg.get("learning_rate", 1e-4))
    lr_lora = float(opt_cfg.get("lr_lora", base_lr))
    lr_decoder = float(opt_cfg.get("lr_decoder", base_lr))
    lr_query = float(opt_cfg.get("lr_query", base_lr))
    lr_mask_head = float(opt_cfg.get("lr_mask_head", base_lr))
    wd = float(opt_cfg.get("weight_decay", 1e-2))

    groups = {
        "lora": {"params": [], "lr": lr_lora, "weight_decay": wd},
        "decoder": {"params": [], "lr": lr_decoder, "weight_decay": wd},
        "query": {"params": [], "lr": lr_query, "weight_decay": wd},
        "mask_head": {"params": [], "lr": lr_mask_head, "weight_decay": wd},
        "other": {"params": [], "lr": base_lr, "weight_decay": wd},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        clean = name[len("module."):] if name.startswith("module.") else name
        if "lora_" in clean:
            groups["lora"]["params"].append(p)
        elif "decoder." in clean or "feature_proj." in clean:
            groups["decoder"]["params"].append(p)
        elif "query_controller." in clean:
            groups["query"]["params"].append(p)
        elif "query_mask_head." in clean:
            groups["mask_head"]["params"].append(p)
        else:
            groups["other"]["params"].append(p)

    param_groups = [g for g in groups.values() if g["params"]]
    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
```

注意：如果已有 optimizer builder 支持分组，不要重写全部；只补齐 query/mask_head 组即可。

---

## 14. 修改 scheduler：支持 poly

文件：

```text
src/train/scheduler.py
```

如果当前只支持 cosine，新增 poly：

```python
from torch.optim.lr_scheduler import LambdaLR

def build_scheduler(optimizer, cfg):
    scfg = cfg.get("scheduler", {}) or {}
    stype = str(scfg.get("type", "cosine")).lower()

    if stype == "poly":
        n_epochs = int((cfg.get("train", {}) or {}).get("n_epochs", 100))
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        power = float(scfg.get("power", 0.9))
        min_lr = float(scfg.get("min_lr", 0.0))

        base_lrs = [group["lr"] for group in optimizer.param_groups]

        def make_lambda(base_lr: float):
            def lr_lambda(epoch: int):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    factor = float(epoch + 1) / float(warmup_epochs)
                else:
                    denom = max(1, n_epochs - warmup_epochs)
                    progress = min(max((epoch - warmup_epochs) / denom, 0.0), 1.0)
                    factor = (1.0 - progress) ** power
                if base_lr > 0:
                    factor = max(min_lr / base_lr, factor)
                return factor
            return lr_lambda

        return LambdaLR(optimizer, lr_lambda=[make_lambda(lr) for lr in base_lrs])

    # keep old cosine behavior
```

---

## 15. temporal augment 支持 `start_epoch`

当前 dataloader 如果不知道 epoch，`start_epoch` 不会生效。最低要求：

- 如果短期不想改 dataloader，就把 temporal augment 的 prob 先设为 0。
- 如果要按最终配置生效，需要 dataloader/transform 能拿到当前 epoch。

推荐简单实现：

1. Dataset 加 `set_epoch(epoch)`。
2. Trainer 每个 epoch 开始时：

```python
if hasattr(train_loader.dataset, "set_epoch"):
    train_loader.dataset.set_epoch(epoch)
```

3. temporal augment 判断：

```python
start_epoch = int(cfg.get("start_epoch", 0))
if self.epoch < start_epoch:
    prob = 0.0
```

---

## 16. 修改 `src/eval/tester.py`

如果 tester 写死旧模型，也要像 trainer 一样使用 `build_model(cfg)`。

推理时必须启用 VidEoMT state：

```python
videomt_state = None
current_video_id = None

for batch in loader:
    ...
    if is_first_window or video_id != current_video_id:
        videomt_state = None
        current_video_id = video_id

    out = model(
        images,
        mode="eval",
        ablation=ablation,
        videomt_state=videomt_state,
        return_videomt_state=True,
    )
    videomt_state = out.get("videomt_state", videomt_state)

    if is_last_window:
        videomt_state = None
        current_video_id = None
```

不要在最终 VidEoMT 测试路径使用 FGM bank。

---

## 17. 不要做的事情

Agent 必须避免这些修改：

1. 不要在 `features` 上新增 gate。
2. 不要把 `query_states.mean(dim=2)` 作为最终路径。
3. 不要新增 CAVIS / DVIS / Mask2Former tracker。
4. 不要保留 bidirectional query 作为默认。
5. 不要把 `residual_alpha` 初始化为非 0 继续 residual。
6. 不要只改配置不改模型；这个配置依赖代码级结构改造。
7. 不要让 query mask head 只作为辅助输出而不影响最终 logits。
8. 不要让每个 window 都从 learned queries 重启；必须跨 window carry `prev_q`。

---

## 18. 最终验收脚本

新增：

```text
tools/check_b23_videomt_final.py
```

内容：

```python
from __future__ import annotations

import torch

from src.utils.config import load_config
from src.models import B23VideoMTWindowModel
from src.losses import VideoMTQueryMaskLoss


def main():
    cfg = load_config("configs/b23_videomt_window_final.yaml")
    model = B23VideoMTWindowModel(cfg).cuda().train()

    x = torch.rand(1, 1, 5, 3, 512, 512, device="cuda")
    y = torch.randint(0, 2, (1, 1, 5, 1, 512, 512), device="cuda").float()

    out = model(x, mode="train", return_videomt_state=True)

    assert "logits" in out
    assert "aux" in out
    assert "videomt_state" in out

    logits = out["logits"]
    aux = out["aux"]

    print("logits:", tuple(logits.shape))
    print("queries:", tuple(aux["videomt_queries"].shape))
    print("query_logits:", tuple(aux["query_logits"].shape))
    print("query_scores:", tuple(aux["query_scores"].shape))

    assert logits.shape == (1, 1, 5, 1, 512, 512)
    assert aux["videomt_queries"].shape[:4] == (1, 1, 5, 32)
    assert aux["query_logits"].shape[:4] == (1, 1, 5, 32)

    criterion = VideoMTQueryMaskLoss(cfg["loss"]).cuda()
    loss, items = criterion(out, y)
    print("loss:", loss.item())
    print(items)

    loss.backward()

    has_query_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in model.named_parameters()
        if "query_controller" in n or "query_mask_head" in n
    )
    assert has_query_grad, "query_controller/query_mask_head has no valid gradients"

    print("OK")


if __name__ == "__main__":
    main()
```

运行：

```bash
python tools/check_b23_videomt_final.py
```

必须通过。

---

## 19. 训练命令

```bash
python train.py --config configs/b23_videomt_window_final.yaml
```

DDP auto torchrun 仍由项目原来的 `maybe_relaunch_with_torchrun` 控制。

---

## 20. 最终结构验收标准

代码完成后，必须满足：

```text
[必须]
- B23VideoMTWindowModel 可以从 src.models 导入。
- loss.type = videomt_query_mask 时使用 VideoMTQueryMaskLoss。
- model.forward 接收 [B,W,T,3,H,W]，输出 [B,W,T,1,512,512]。
- aux 中包含 videomt_queries/query_logits/query_scores。
- decoder 输出 features128。
- query logits 参与最终 logits，不只是 aux。
- DINOv3 forward_with_queries 中 query tokens 进入最后 4 个 blocks。
- 跨 window 可通过 videomt_state["prev_q"] 传递 query。
- 配置中的 residual.enabled=false 后，模型不再使用 residual_alpha。
- bidirectional=false 后，模型不再运行 backward query。
- `tools/check_b23_videomt_final.py` 能 forward + loss + backward。

[建议]
- 打印一次 debug shapes。
- 训练第一个 epoch 后确认 query_mask loss 非 0 且有限。
- 确认 query_controller/query_mask_head 参数在 optimizer param group 内。
- 确认 LoRA 只在最后 4 层或目标配置指定层启用。
```

---

## 21. 给 Agent 的一句话总目标

请把 `b23_videomt_window` 改成 **VidEoMT-like encoder-query-mask final model**：  
`Linear(prev_q)+Q_lrn` 产生 query，query 进入 DINOv3 最后 4 个 block，与 patch token 同步交互；decoder 导出 f128；query mask head 用 query 直接预测 mask 并聚合成最终 logits；loss 同时监督最终 logits 和 query-level masks；训练/验证/测试都支持跨 window 的 query state。不要使用 gate，不要使用 `mean(query)` residual，不要引入重型 tracker。
