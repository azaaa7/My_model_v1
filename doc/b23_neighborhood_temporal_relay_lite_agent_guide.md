# My_model_v1：DINOv3-LoRA + Neighborhood Temporal Relay Lite 全量代码修改指导

> 适用仓库：`https://github.com/azaaa7/My_model_v1`  
> 目标配置：`configs/b23_temporal_relay_lite.yaml`  
> 推荐训练命令：
>
> ```bash
> bash scripts/train_ddp.sh configs/b23_temporal_relay_lite.yaml
> ```
>
> 本文档面向自动编码 Agent。Agent 必须逐文件执行修改、运行结构与训练验收，并在最终回复中列出实际改动、完整测试结果和未完成事项。

---

## 0. 本次修正的核心

上一版只强调 relay token 的跨帧信息交换，没有明确要求 patch token 本身进行具有运动容忍度的邻域交互。本版必须修正为：

```text
当前 patch (t,h,w)
不只与其他帧的 (t±1,h,w) 交换，
而是与相邻帧中以 (h,w) 为中心的 3×3 空间邻域交换。
```

默认邻域：

```text
时间范围：t-1, t, t+1
空间范围：h-1:h+1, w-1:w+1
候选 token 数：3 × 3 × 3 = 27
```

因此最终时序模块不是“同位置时间注意力”，而是：

```text
局部时空邻域注意力
+ 少量全局 relay token
```

LoRA 同时修改为：

```yaml
rank: 32
alpha: 64
```

---

## 1. 最终目标结构

将旧的 VidEoMT-like 路径：

```text
32 queries
→ query 注入 DINOv3 最后 4 个 block
→ Linear(prev_q)+Q_lrn
→ QueryMaskHead
→ connected-component Hungarian loss
```

替换为：

```text
视频 clip [B,W,T,3,512,512]
        │
        ▼
DINOv3 原生 forward
+ LoRA blocks 16-23
+ rank=32, alpha=64
        │
        ▼
patch features [B*W,T,1024,32,32]
        │
        ▼
Motion-tolerant Local Spatiotemporal Attention
每个 patch 关注：
  时间 t-1,t,t+1
  空间 3×3 邻域
  共最多 27 个候选 token
        │
        ▼
2-token Global Temporal Relay
  1. relay 从每帧局部增强后的 patch 中吸收信息
  2. relay 跨帧交换全局信息
  3. relay 回注到每帧 patch
        │
        ▼
LayerNorm + concat + projection
无 gate、无 alpha residual scale
        │
        ▼
原有 feature_proj
        │
        ▼
原有 LiteBoundaryDecoder
源码和网络结构不改
        │
        ▼
mask logits + boundary128
```

验证和测试：

```text
原始分辨率视频
→ 相同时间 clip 使用相同 tile 坐标
→ 512×512 tile 推理
→ Hann 加权拼接 logits
→ 重叠 clip 按真实 frame index 平均 logits
→ 原始 H×W 上计算 IoU/F1
```

---

## 2. 不可违反的约束

### 2.1 禁止事项

Agent 不得：

1. 修改 `src/models/decoders/lite_boundary_decoder.py` 的结构。
2. 引入 Mask2Former、FPN、U-Net、DVIS、CAVIS 等大型 decoder。
3. 使用 `QueryMaskHead` 作为最终输出。
4. 使用 `VideoMTQueryController`。
5. 使用 `Linear(prev_q)+Q_lrn`。
6. 使用跨 window recurrent query state。
7. 使用 connected-component Hungarian matching。
8. 使用 query BCE、query Dice 或 no-object loss。
9. 向 DINOv3 原生 token 序列中插入 query/relay token。
10. 只做同一空间坐标的 temporal attention。
11. 使用零初始化 gate。
12. 使用必须手工调大的全局标量 gate。
13. 使用 `alpha * temporal_delta` 形式的可学习全局缩放。
14. 新增 SRM、DCT、wavelet、光流、帧差或高通分支。
15. 验证或测试时将完整原图 resize 到 512。
16. 删除旧模型、旧配置和旧验收脚本。

### 2.2 必须保留

1. DINOv3 `dinov3_vitl16`。
2. 现有权重加载逻辑。
3. DINOv3 LoRA 微调。
4. LoRA blocks 16-23。
5. LoRA targets `attn.qkv,attn.proj`。
6. LoRA rank 32。
7. LoRA alpha 64。
8. 现有 `LiteBoundaryDecoder`。
9. 最终 logits 来自 `dec_out["logits"]`。
10. `boundary128` 辅助监督。
11. 原始分辨率验证和测试。
12. 当前 DDP、AMP、checkpoint 和 activation checkpoint 框架。

---

## 3. 新模型与文件规划

### 3.1 新增文件

```text
configs/b23_temporal_relay_lite.yaml
src/models/modules/neighborhood_temporal_relay.py
src/models/b23_temporal_relay_lite_model.py
src/eval/original_resolution.py
tools/check_b23_temporal_relay_lite.py
```

### 3.2 修改文件

```text
src/models/modules/__init__.py
src/models/__init__.py
src/models/builder.py
src/train/optimizer.py
src/train/trainer.py
src/data/dataset.py
src/data/transforms.py
src/eval/tester.py
test.py
```

按实际项目结构可修改：

```text
src/data/__init__.py
src/eval/__init__.py
```

### 3.3 不允许修改

```text
src/models/decoders/lite_boundary_decoder.py
src/models/b23_videomt_window_model.py
src/models/videomt/*
src/losses/videomt_query_mask_loss.py
configs/b23_videomt_window_final.yaml
tools/check_b23_videomt_final.py
```

---

## 4. 配置文件

新增或覆盖：

```text
configs/b23_temporal_relay_lite.yaml
```

```yaml
type: train
seed: 666666

model:
  name: "B23TemporalRelayLiteModel"
  version: "dinov3_lora_neighborhood_temporal_relay_lite"

input_size: 512
batch_size: 1
num_clips: 1
num_frames: 5
clip_stride: 1
encoder_chunk: 2
gt_ratio: 1

train_full_video_windows: false
train_max_windows_per_video: 0

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

augment_prob: 0.30
spatial_augment_prob: 0.30
appearance_augment_prob: 0.20

augmentation:
  random_scale_limit: [-0.2, 0.1]
  same_spatial_transform_for_clip: true

temporal_augment:
  frame_swap:
    enabled: false
    prob: 0.0
  frame_drop:
    enabled: false
    prob: 0.0

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
    enabled: false

lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.05
  layers: [16, 17, 18, 19, 20, 21, 22, 23]
  targets: "attn.qkv,attn.proj"

temporal_relay:
  enabled: true
  dim: 1024

  local_neighborhood:
    enabled: true
    temporal_radius: 1
    spatial_radius: 1
    spatial_dilation: 1
    include_current_frame: true
    include_current_patch: true
    boundary_mode: "mask"
    num_heads: 8
    qkv_bias: true
    attn_dropout: 0.0
    proj_dropout: 0.0
    relative_position_bias: true
    relative_position_type: "learned_3d"
    use_sdpa: true

  global_relay:
    enabled: true
    num_tokens: 2
    num_layers: 2
    num_heads: 8
    ffn_ratio: 4.0
    dropout: 0.0
    temporal_rope: true
    relay_identity_embedding: true

  recurrent_state: false
  fusion: "concat_projection"
  debug_stats: true
  finite_check: false

tfcu:
  version: "neighborhood_temporal_relay_lite"

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
    enabled: false
  mask128_head:
    enabled: false
  mask256_head:
    enabled: true
  boundary_head:
    enabled: true
    resolution: 128

loss:
  type: "segmentation"
  bce:
    enabled: true
    weight: 1.0
  dice:
    enabled: true
    weight: 1.0
    smooth: 1.0e-6

aux_loss:
  boundary128:
    enabled: true
    weight: 1.0

optimizer:
  type: "adamw"
  learning_rate: 1.0e-4
  lr_lora: 2.0e-5
  lr_temporal: 1.0e-4
  lr_decoder: 1.0e-4
  weight_decay: 5.0e-2
  betas: [0.9, 0.999]
  eps: 1.0e-8

scheduler:
  type: "cosine"
  warmup_epochs: 2
  min_lr: 5.0e-7

val_full_video: true
val_test_max_clips: 9999
val_num_workers: 0

test_full_video: true
test_max_clips: 9999

inference:
  original_resolution: true
  tile_size: 512
  tile_stride: 384
  tile_batch_size: 1
  blending: "hann"
  hann_min_weight: 1.0e-3
  clip_stride: 2
  average_overlapping_logits: true
  pad_mode: "replicate"

train:
  n_epochs: 200
  save_dir: "runs/b23_temporal_relay_lite"
  val_interval: 5
  max_grad_norm: 1.0
  skip_nonfinite: true

checkpoint:
  save_trainable_only: true
  save_optimizer: true
  save_scheduler: true

ema:
  enabled: false

stability:
  logit_clamp: 30.0
  nan_to_num: true

ddp:
  auto_torchrun: true
  cuda_visible_devices: "4,5"
  nproc_per_node: 2
  dist_backend: "nccl"
  timeout_minutes: 120
  find_unused_parameters: false
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  torchrun_log_dir: "runs/b23_temporal_relay_lite/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
  log_temporal_relay_stats: true
  max_train_steps: 0
```

### 4.1 配置硬性验收

```python
assert cfg["model"]["name"] == "B23TemporalRelayLiteModel"

assert cfg["lora"]["rank"] == 32
assert cfg["lora"]["alpha"] == 64

assert cfg["lora"]["layers"] == [
    16,17,18,19,20,21,22,23
]

local_cfg = cfg["temporal_relay"]["local_neighborhood"]

assert local_cfg["temporal_radius"] == 1
assert local_cfg["spatial_radius"] == 1
assert local_cfg["relative_position_bias"] is True

assert cfg["temporal_relay"]["global_relay"]["num_tokens"] == 2
assert cfg["temporal_relay"]["recurrent_state"] is False

assert cfg["decoder"]["mask256_head"]["enabled"] is True
assert cfg["inference"]["original_resolution"] is True
```

---

## 5. 邻域时空注意力的理论定义

输入 DINOv3 patch feature：

```text
X ∈ R[B,T,C,H,W]
H=W=32
C=1024
```

对当前 token：

```text
x(t,h,w)
```

构造邻域：

```text
dt ∈ {-1,0,+1}
dy ∈ {-1,0,+1}
dx ∈ {-1,0,+1}
```

候选 key/value：

```text
x(t+dt, h+dy, w+dx)
```

最大候选数量：

```text
K = 3 × 3 × 3 = 27
```

公式：

```text
q(t,h,w) = Wq · LN(x(t,h,w))

k(dt,dy,dx) =
  Wk · LN(x(t+dt,h+dy,w+dx))

v(dt,dy,dx) =
  Wv · LN(x(t+dt,h+dy,w+dx))
```

注意力：

```text
A = softmax(
      qk^T / sqrt(d)
      + Brel(dt,dy,dx)
      + invalid_mask
    )
```

输出：

```text
Δx_local(t,h,w) = Σ A · v
```

更新：

```text
X_local = X + Proj(ΔX_local)
X_local = X_local + FFN(LN(X_local))
```

这里是标准 Transformer residual，不使用 gate。

### 5.1 为什么必须使用邻域而不是同位置

同一物体或篡改区域在相邻帧中可能发生：

- 相机运动；
- 目标运动；
- mask 边界移动；
- 视频修复内容传播偏移；
- patch 网格量化误差。

因此：

```text
(t,h,w)
```

不一定对应下一帧的：

```text
(t+1,h,w)
```

允许搜索 `3×3` 邻域后，模型可在相邻帧中选择：

```text
(h-1,w-1) ... (h+1,w+1)
```

作为对应信息。

在 patch size 为 16 时，`spatial_radius=1` 对应约 ±16 像素的离散运动容忍范围。

---

## 6. 新时序模块文件

新增：

```text
src/models/modules/neighborhood_temporal_relay.py
```

### 6.1 推荐内部类

```python
class FeedForward(nn.Module)
class LearnedRelativePositionBias3D(nn.Module)
class LocalSpatiotemporalNeighborhoodAttention(nn.Module)
class TemporalRotaryEmbedding(nn.Module)
class RelayTemporalSelfAttention(nn.Module)
class GlobalTemporalRelayLayer(nn.Module)
class NeighborhoodTemporalRelayFusion(nn.Module)
```

### 6.2 对外接口

```python
class NeighborhoodTemporalRelayFusion(nn.Module):
    def __init__(self, cfg: dict):
        ...

    def forward(
        self,
        features: torch.Tensor,
        disable_temporal: bool = False,
        disable_local_neighborhood: bool = False,
        disable_global_relay: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        # features: [B,T,C,H,W]
        # return:
        #   fused_features: [B,T,C,H,W]
        #   debug: dict
        ...
```

---

## 7. LocalSpatiotemporalNeighborhoodAttention

### 7.1 输入输出

输入：

```text
[B,T,C,H,W]
```

输出：

```text
[B,T,C,H,W]
```

不能改变 T/H/W。

### 7.2 不允许全局时空注意力

禁止将全部：

```text
T×H×W = 5×32×32 = 5120
```

个 token 做完整 self-attention。

必须只对每个 query token 的局部候选进行注意力，候选数默认 27。

### 7.3 邻域提取

推荐使用 `torch.nn.functional.unfold` 提取每帧空间邻域。

先重排：

```python
x_bt = x.reshape(B * T, C, H, W)
```

提取 3×3：

```python
patches = F.unfold(
    x_bt,
    kernel_size=3,
    dilation=1,
    padding=1,
    stride=1,
)
```

输出：

```text
[B*T, C*9, H*W]
```

恢复：

```text
[B,T,H*W,9,C]
```

然后根据 temporal radius 堆叠：

```text
t-1 的 9 个
t   的 9 个
t+1 的 9 个
```

得到：

```text
neighbors [B,T,H*W,27,C]
```

### 7.4 时间边界处理

配置：

```yaml
boundary_mode: "mask"
```

对于：

```text
t=0 的 t-1
t=T-1 的 t+1
```

不得简单复制有效帧并当作真实候选。

必须生成：

```text
valid_neighbor_mask [T,27]
```

无效候选在 attention logits 上加：

```python
torch.finfo(logits.dtype).min
```

或稳定的足够小负数。

禁止产生全 invalid 行。因为 `dt=0` 且当前位置始终有效。

### 7.5 空间边界处理

`F.unfold(padding=1)` 会产生零填充位置。

必须同步生成空间 validity mask：

```python
ones = torch.ones(
    1,1,H,W,
    device=x.device,
    dtype=x.dtype,
)

valid_spatial = F.unfold(
    ones,
    kernel_size=3,
    padding=1,
).reshape(1,H*W,9)
```

将空间无效位置加入 attention mask。

不能让图像边缘将零 padding 当成真实特征参与注意力。

### 7.6 Q/K/V 投影

```python
self.q_proj = nn.Linear(C, C, bias=qkv_bias)
self.k_proj = nn.Linear(C, C, bias=qkv_bias)
self.v_proj = nn.Linear(C, C, bias=qkv_bias)
self.out_proj = nn.Linear(C, C)
```

输入先 LayerNorm：

```python
query = self.q_proj(
    self.query_norm(center_tokens)
)

keys = self.k_proj(
    self.kv_norm(neighbors)
)

values = self.v_proj(
    self.kv_norm(neighbors)
)
```

### 7.7 多头形状

```text
Q: [B,T,N,heads,1,head_dim]
K: [B,T,N,heads,27,head_dim]
V: [B,T,N,heads,27,head_dim]
```

注意力 logits：

```python
logits = (
    query * keys
).sum(dim=-1) * self.scale
```

形状：

```text
[B,T,N,heads,27]
```

### 7.8 3D 相对位置偏置

必须实现可学习偏置：

```text
Brel(dt,dy,dx)
```

默认范围：

```text
dt ∈ [-1,1]
dy ∈ [-1,1]
dx ∈ [-1,1]
```

偏置表：

```python
self.relative_bias = nn.Parameter(
    torch.zeros(
        num_heads,
        3,
        3,
        3,
    )
)
```

初始化允许：

```python
nn.init.trunc_normal_(
    self.relative_bias,
    std=0.02,
)
```

不能把整个 local attention 输出零初始化。

构建固定索引，将偏置变为：

```text
[heads,27]
```

加到 logits：

```python
logits = logits + relative_bias[
    None,None,None,:,:,
]
```

### 7.9 Attention 计算

可直接：

```python
attn = torch.softmax(
    logits.float(),
    dim=-1,
).to(logits.dtype)

attn = self.attn_dropout(attn)

out = torch.sum(
    attn.unsqueeze(-1) * values,
    dim=-2,
)
```

softmax 推荐使用 float32 以提高 AMP 稳定性。

若使用 SDPA，必须确认支持每个 query 的 27 个显式 local key/value，不能退化为全局 attention。

### 7.10 Local residual

```python
local_delta = self.out_proj(out)
local_tokens = center_tokens + local_delta
local_tokens = local_tokens + self.ffn(
    self.ffn_norm(local_tokens)
)
```

禁止：

```python
center_tokens + gate * local_delta
center_tokens + alpha * local_delta
```

### 7.11 Local debug

必须返回：

```python
local_debug = {
    "local_candidate_count": 27,
    "local_temporal_radius": 1,
    "local_spatial_radius": 1,
    "local_delta_ratio": float(
        local_delta.float().norm()
        / (center_tokens.float().norm() + 1e-6)
    ),
    "local_attention_entropy": float(...),
    "local_attention_max": float(...),
}
```

---

## 8. Global Temporal Relay

Local neighborhood attention 负责：

```text
短程运动和邻域对应
```

Global relay 负责：

```text
帧级和 clip 级全局上下文
```

### 8.1 relay token

每帧 2 个：

```python
self.relay_tokens = nn.Parameter(
    torch.empty(1,1,2,C)
)

nn.init.trunc_normal_(
    self.relay_tokens,
    std=0.02,
)
```

### 8.2 从 local tokens 吸收信息

```python
relay = relay + CrossAttention(
    query=LN(relay),
    key=LN(local_tokens),
    value=LN(local_tokens),
)

relay = relay + FFN(LN(relay))
```

形状：

```text
query: [B*T,2,C]
key/value: [B*T,H*W,C]
```

### 8.3 relay 跨时间交换

拼接为：

```text
[B,T*2,C]
```

加入：

- temporal RoPE；
- relay identity embedding。

```python
relay = relay + relay_identity_embedding[
    relay_index
]

relay = relay + temporal_self_attention(
    LN(relay),
    time_index=time_index,
)

relay = relay + FFN(LN(relay))
```

### 8.4 relay 回注 local tokens

```python
relay_delta = CrossAttention(
    query=LN(local_tokens),
    key=LN(relay),
    value=LN(relay),
)

global_tokens = local_tokens + relay_delta
global_tokens = global_tokens + FFN(
    LN(global_tokens)
)
```

仍然禁止 gate。

---

## 9. 最终无门控融合

保留原始空间 token：

```text
X_spatial
```

局部邻域与 relay 处理后的 token：

```text
X_temporal
```

最终：

```python
spatial = self.spatial_norm(
    spatial_tokens
)

temporal = self.temporal_norm(
    temporal_tokens
)

fused = self.fusion_proj(
    torch.cat(
        [spatial, temporal],
        dim=-1,
    )
)
```

推荐：

```python
self.fusion_proj = nn.Sequential(
    nn.Linear(2 * C, C),
    nn.GELU(),
    nn.LayerNorm(C),
)
```

初始化：

```python
nn.init.xavier_uniform_(
    self.fusion_proj[0].weight
)

nn.init.zeros_(
    self.fusion_proj[0].bias
)
```

不允许新增：

```python
sigmoid gate
scalar alpha
per-branch zero scale
```

---

## 10. 多层更新顺序

默认：

```yaml
global_relay:
  num_layers: 2
```

推荐每层执行：

```text
1. local neighborhood attention
2. relay 从 local tokens 吸收
3. relay temporal attention
4. relay 回注 local tokens
```

第二层使用第一层输出。

若显存过高，允许：

```text
只执行一次 local neighborhood attention
+ 两层 global relay
```

但第一版优先完整执行两层 local-global block。

Agent 必须在 debug 中打印实际策略：

```text
local_per_layer=true/false
```

---

## 11. 消融开关

必须支持：

```python
ablation = {
    "disable_temporal_relay": False,
    "disable_local_neighborhood": False,
    "disable_global_relay": False,
}
```

### 11.1 全部禁用

```python
disable_temporal_relay=True
```

直接返回 DINOv3 空间特征，不经过 concat projection。

### 11.2 仅关闭 local

```python
disable_local_neighborhood=True
```

保留 global relay。

### 11.3 仅关闭 relay

```python
disable_global_relay=True
```

保留 local 3D neighborhood attention。

这样可以分别验证：

```text
局部时空邻域贡献
全局 relay 贡献
```

---

## 12. 新模型

新增：

```text
src/models/b23_temporal_relay_lite_model.py
```

类名：

```python
class B23TemporalRelayLiteModel(nn.Module):
```

### 12.1 初始化

```python
self.encoder = DINOv3B23Encoder(
    dinov3_cfg,
    lora_cfg,
)

self.temporal_relay = (
    NeighborhoodTemporalRelayFusion(
        temporal_relay_cfg
    )
)

self.feature_proj = nn.Sequential(
    nn.GroupNorm(32, 1024),
    nn.Conv2d(
        1024,
        decoder_in_channels,
        kernel_size=1,
        bias=False,
    ),
    nn.GroupNorm(
        8,
        decoder_in_channels,
    ),
    nn.GELU(),
)

self.decoder = LiteBoundaryDecoder(
    decoder_cfg
)
```

禁止实例化：

```text
VideoMTQueryController
QueryMaskHead
WindowQueryFusion
```

### 12.2 forward

```python
def forward(
    self,
    video: torch.Tensor,
    mode: str | None = None,
    ablation: dict | None = None,
    epoch: int | None = None,
    **kwargs,
) -> dict:
```

支持：

```text
[B,T,3,512,512]
[B,W,T,3,512,512]
```

5D 输入补 window 维：

```python
video = video[:, None]
```

### 12.3 DINO 编码

只调用：

```python
self.encoder(frames)
```

不得调用：

```python
forward_with_queries
```

编码器输出恢复为：

```text
[B*W,T,1024,32,32]
```

### 12.4 时序模块调用

```python
ablation = ablation or {}

fused_features, temporal_debug = (
    self.temporal_relay(
        patch_features,
        disable_temporal=bool(
            ablation.get(
                "disable_temporal_relay",
                False,
            )
        ),
        disable_local_neighborhood=bool(
            ablation.get(
                "disable_local_neighborhood",
                False,
            )
        ),
        disable_global_relay=bool(
            ablation.get(
                "disable_global_relay",
                False,
            )
        ),
    )
)
```

### 12.5 Decoder

```python
flat_fused = fused_features.reshape(
    B * W * T,
    C,
    32,
    32,
)

dec_in = self.feature_proj(
    flat_fused
)

dec_out = self.decoder(
    dec_in
)
```

最终 logits：

```python
logits = dec_out["logits"]
```

不得由 query 聚合产生。

### 12.6 输出

```python
out = {
    "logits": logits,
    "aux": {
        "mask128": restored_mask128,
        "mask256": restored_mask256,
        "boundary128": restored_boundary128,
        "edge_logits": restored_boundary128,
        "ccm_mask32": None,
        "fgm_mask32": None,
        "fgm_cue": None,
        "debug": {
            **temporal_debug,
            "patch_features_shape": tuple(
                patch_features.shape
            ),
            "fused_features_shape": tuple(
                fused_features.shape
            ),
            "decoder_debug": dec_out.get(
                "debug", {}
            ),
        },
    },
}
```

不得输出：

```text
query_logits
query_scores
videomt_queries
videomt_state
```

---

## 13. LoRA 验收

配置必须为：

```yaml
rank: 32
alpha: 64
```

Agent 必须验证实际模块参数，而不仅检查 YAML。

打印 LoRA 模块的：

```text
rank
alpha
scaling = alpha / rank
```

预期：

```text
rank=32
alpha=64
scaling=2.0
```

检查参数层：

```python
assert any(
    "blocks.16." in name
    for name in lora_names
)

assert any(
    "blocks.23." in name
    for name in lora_names
)

assert not any(
    "blocks.0." in name
    for name in lora_names
)

assert not any(
    "mlp.fc1" in name
    for name in lora_names
)

assert not any(
    "mlp.fc2" in name
    for name in lora_names
)
```

一次 backward 后必须存在 finite LoRA 梯度。

---

## 14. Optimizer

修改：

```text
src/train/optimizer.py
```

新增独立组：

```python
"temporal_relay": {
    "params": [],
    "lr": float(
        opt_cfg.get(
            "lr_temporal",
            base_lr,
        )
    ),
},
```

匹配顺序：

```python
if "lora_" in name:
    group = "lora"

elif "temporal_relay." in name:
    group = "temporal_relay"

elif (
    "decoder." in name
    or "feature_proj." in name
):
    group = "decoder"
```

预期学习率：

```text
LoRA             2e-5
Neighborhood     1e-4
Global relay     1e-4
Feature projection 1e-4
Decoder          1e-4
```

`Neighborhood` 和 `Global relay` 均属于 `temporal_relay` group。

必须检查每个 trainable 参数恰好出现一次。

---

## 15. Loss

不新增 loss。

使用：

```text
BCE + Dice + boundary128
```

配置：

```yaml
loss:
  bce:
    enabled: true
    weight: 1.0
  dice:
    enabled: true
    weight: 1.0

aux_loss:
  boundary128:
    enabled: true
    weight: 1.0
```

禁止 query loss。

---

## 16. 数据增强

同一 clip 的所有帧必须共享：

```text
crop
scale
flip
rotation
```

必须关闭：

```text
frame_swap
frame_drop
```

RandomScale：

```yaml
random_scale_limit: [-0.2,0.1]
```

不得继续允许缩小至原图 20%。

---

## 17. 原始分辨率验证和测试

保持上一版要求：

1. val/test 不整体 resize 到 512。
2. tile size 512。
3. stride 384。
4. 同一 clip 所有帧使用完全相同 tile 坐标。
5. tile logits 使用 Hann 权重拼接。
6. 重叠 clip 按真实 frame index 平均 logits。
7. 每个真实 frame 只计一次指标。
8. 指标在原始 H×W 上计算。

### 17.1 tile 内邻域注意力说明

局部时空邻域是在每个 512 tile 的 DINO patch grid 内计算。

tile overlap 为 128 像素，能降低 tile 边缘缺失邻域信息的问题。

Hann 权重会进一步降低 tile 边缘预测对最终结果的影响。

---

## 18. 原始分辨率推理文件

新增：

```text
src/eval/original_resolution.py
```

必须实现：

```python
make_tile_starts(...)
make_hann_weight(...)
tiled_clip_logits(...)
evaluate_original_resolution(...)
```

完整规则沿用：

```text
replicate padding
float32 logit accumulation
Hann min weight 1e-3
crop back to original H×W
average logits before sigmoid
```

synthetic 验收：

```python
clip = torch.rand(
    5,3,544,768,
    device=device,
)

logits = tiled_clip_logits(
    model,
    clip,
    cfg,
)

assert logits.shape == (
    5,1,544,768
)
```

---

## 19. Trainer 与 Tester

### 19.1 Trainer

新模型走普通无状态 forward：

```python
out = model(
    images,
    mode="train",
    ablation=ablation,
    epoch=epoch,
)
```

不得进入 VidEoMT state 分支。

### 19.2 原始分辨率 evaluate 分流

当：

```text
model.name == B23TemporalRelayLiteModel
inference.original_resolution == true
```

调用：

```python
evaluate_original_resolution(...)
```

### 19.3 SyncBatchNorm

DDP 下、optimizer 构建前：

```python
model = (
    torch.nn.SyncBatchNorm
    .convert_sync_batchnorm(model)
)
```

不修改 decoder 源码。

### 19.4 test CLI

新增：

```text
--disable-temporal-relay
--disable-local-neighborhood
--disable-global-relay
```

---

## 20. Debug 指标

必须输出：

```text
local_candidate_count
local_temporal_radius
local_spatial_radius
local_delta_ratio
local_attention_entropy
local_attention_max

relay_token_norm
relay_delta_ratio
relay_attention_entropy

temporal_output_std
temporal_relay_disabled
local_neighborhood_disabled
global_relay_disabled
```

参考报警：

```text
local_delta_ratio < 1e-4
```

局部时空分支可能未生效。

```text
local_attention_entropy 接近 log(27)
且长期不下降
```

可能注意力近似均匀，未学到匹配关系。

```text
local_attention_max > 0.99
```

可能过度塌缩到单一邻居。

只报警，不自动加入 gate。

---

## 21. 验收脚本

新增：

```text
tools/check_b23_temporal_relay_lite.py
```

### 21.1 Forward

输入：

```python
x = torch.rand(
    1,1,5,3,512,512,
    device=device,
)
```

输出：

```text
logits [1,1,5,1,512,512]
boundary128 [1,1,5,1,128,128]
```

### 21.2 Local candidate count

必须断言：

```python
assert debug[
    "local_candidate_count"
] == 27
```

### 21.3 LoRA 参数

必须断言：

```python
assert actual_lora_rank == 32
assert actual_lora_alpha == 64
```

不能只检查 YAML。

### 21.4 Gradient

以下模块必须有 finite 梯度：

```text
LoRA
local neighborhood q/k/v/out projections
3D relative position bias
global relay
fusion projection
feature_proj
decoder
```

### 21.5 消融验收

分别执行：

```python
disable_temporal_relay=True
disable_local_neighborhood=True
disable_global_relay=True
```

检查 shape 不变，并确认 debug flag 正确。

### 21.6 邻域交换功能测试

构造简单 feature 测试，不经过 DINO：

```text
中心帧中心 patch 为 0
下一帧右侧邻居 patch 为高值
其他 patch 为低值
```

运行 local neighborhood attention 后，中心 patch 输出必须受到右侧邻居影响。

测试重点不是固定数值，而是：

```python
assert not torch.allclose(
    output_center,
    input_center,
)
```

并确认被设置高值的邻居属于有效 3×3 候选。

再构造距离超过 radius=1 的 patch，高值不得进入该 query 的候选集合。

### 21.7 原始分辨率

```text
544×768 → 544×768
```

且 finite。

---

## 22. 训练前运行顺序

### 22.1 语法检查

```bash
python -m compileall \
  src/models/modules/neighborhood_temporal_relay.py \
  src/models/b23_temporal_relay_lite_model.py \
  src/eval/original_resolution.py \
  tools/check_b23_temporal_relay_lite.py
```

### 22.2 单元与结构验收

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_b23_temporal_relay_lite.py
```

### 22.3 20-step smoke train

配置：

```yaml
debug:
  max_train_steps: 20
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --config configs/b23_temporal_relay_lite.yaml
```

完成后恢复：

```yaml
max_train_steps: 0
```

### 22.4 DDP

```bash
bash scripts/train_ddp.sh \
  configs/b23_temporal_relay_lite.yaml
```

### 22.5 原始分辨率测试

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --config configs/b23_temporal_relay_lite.yaml \
  --checkpoint \
  runs/b23_temporal_relay_lite/best_iou.pt
```

---

## 23. 必做消融

### A. DINOv3-LoRA baseline

```yaml
temporal_relay:
  enabled: false
```

### B. 仅局部时空邻域

```yaml
local_neighborhood:
  enabled: true
global_relay:
  enabled: false
```

### C. 仅全局 relay

```yaml
local_neighborhood:
  enabled: false
global_relay:
  enabled: true
```

### D. 推荐完整模型

```yaml
local_neighborhood:
  enabled: true
  temporal_radius: 1
  spatial_radius: 1

global_relay:
  enabled: true
  num_tokens: 2
```

### E. 可选更大邻域

```yaml
spatial_radius: 2
```

候选数：

```text
3 × 5 × 5 = 75
```

仅作为消融，不作为默认配置。

记录：

```text
原始分辨率 IoU
原始分辨率 F1
Precision
Recall
DVI/CPNET
OPN
最大显存
每帧耗时
```

---

## 24. 完成标准

```text
[ ] 新模型可由 build_model 构建
[ ] 不调用 DINO forward_with_queries
[ ] 不实例化 VideoMTQueryController
[ ] 不实例化 QueryMaskHead
[ ] LoRA rank 实际为 32
[ ] LoRA alpha 实际为 64
[ ] LoRA scaling 实际为 2.0
[ ] LoRA 注入 blocks 16-23
[ ] LoRA 只作用于 qkv/proj
[ ] DINOv3 非 LoRA 参数冻结
[ ] local temporal radius=1
[ ] local spatial radius=1
[ ] 每个 patch 最大候选数=27
[ ] 当前 patch 可关注邻帧周围 3×3 patch
[ ] 超出 radius 的 patch 不进入候选
[ ] 使用 learned 3D relative position bias
[ ] local attention 输入输出 shape 一致
[ ] global relay token 数=2
[ ] global relay 层数=2
[ ] 不存在零初始化 gate
[ ] 不存在 scalar alpha residual gate
[ ] 不存在 recurrent window state
[ ] final logits 来自 LiteBoundaryDecoder
[ ] LiteBoundaryDecoder 源码未修改
[ ] mask256_head 开启
[ ] BCE/Dice/boundary loss finite
[ ] LoRA 梯度 finite
[ ] local attention 梯度 finite
[ ] relative bias 梯度 finite
[ ] relay 梯度 finite
[ ] feature_proj 梯度 finite
[ ] decoder 梯度 finite
[ ] optimizer 有独立 temporal_relay group
[ ] val/test 不整体 resize 到 512
[ ] 原始分辨率同步 tiled inference
[ ] Hann 加权拼接
[ ] 重叠 clip 按 frame index 平均 logits
[ ] 每个真实 frame 只统计一次
[ ] 544×768 输出保持 544×768
[ ] 20-step smoke train 通过
[ ] DDP 首个 epoch 通过
[ ] 原始分辨率 val/test 通过
```

---

## 25. Agent 最终回复格式

### 修改文件

```text
新增：
- ...

修改：
- ...

未修改但核查：
- src/models/decoders/lite_boundary_decoder.py
- ...
```

### 结构确认

```text
DINO forward:
LoRA blocks:
LoRA rank:
LoRA alpha:
Local temporal radius:
Local spatial radius:
Candidates per patch:
Relative position bias:
Relay tokens:
Relay layers:
Fusion:
Decoder:
Original-resolution inference:
```

### 验收结果

必须提供：

```text
compileall
forward shape
candidate count=27
LoRA rank=32
LoRA alpha=64
optimizer groups
gradient checks
neighborhood functional test
544×768 tiled inference
20-step smoke train
first original-resolution val IoU/F1
```

任何未运行测试必须明确说明。

---

## 26. 最终结构摘要

最终提交必须是：

```text
DINOv3 native patch features
+ LoRA blocks 16-23
+ rank 32 / alpha 64
+ local spatiotemporal 3×3×3 neighborhood attention
+ learned 3D relative position bias
+ 2-token global temporal relay
+ no gate
+ no recurrent state
+ concat-projection fusion
+ unchanged LiteBoundaryDecoder
+ BCE + Dice + boundary loss
+ original-resolution tiled validation/test
```

核心思想：

1. DINOv3 提供强空间表征。
2. LoRA rank 32 / alpha 64 提供足够任务适配能力。
3. 每个 patch 在相邻帧中搜索周围 patch，而不是假设同坐标严格对应。
4. 局部时空注意力处理运动与边界位移。
5. 少量 relay token 补充 clip 级全局信息。
6. 不使用手工取证分支。
7. 不修改轻量 decoder。
8. 不使用零初始化 gate。
9. 所有最终指标在原始分辨率上计算。
