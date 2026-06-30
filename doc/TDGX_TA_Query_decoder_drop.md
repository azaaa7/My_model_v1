# B24 Aggressive-B 实现指导文档：TDGX + Temporal-Only Attention + Query Volume Decoder + TubeDrop

## 0. 目标

本方案用于在 `b24_dinov3_iml_video_paperloss_lora32.yml` baseline 上做激进增强，目标是：

1. 最大程度提高 DVI / CPNET 等同源验证集精度。
2. 尽量保持 baseline 的跨源泛化能力。
3. 不引入光流网络。
4. 不引入频域分支。
5. 不引入音频或其它模态。
6. 不引入第二个大型视觉 backbone。
7. loss 保持少而必要，默认只使用 3 个 loss term，最多保留 4 个。

最终模型建议命名：

```text
B24DINOv3IMLTDGXTOAttnQVolVideoModel
```

配置文件建议命名：

```text
configs/b24_dinov3_iml_tdgx_toattn_qvol_tubedrop_video_paperloss_lora32.yml
```

核心结构：

```text
DINOv3 frozen + LoRA
        ↓
DINO features [B, M, K, 1024, 32, 32]
        ↓
Temporal Tube Dropout      # train only, no gate
        ↓
TDGX                       # temporal-difference gated residual injection
        ↓
Temporal-Only Attention    # no gate, residual + zero-init
        ↓
Query Volume Decoder       # no gate, query-based 3D mask volume
        ↓
Final mask logits [B, M, K, 1, 512, 512]
```

------

## 1. 总体原则

### 1.1 哪些模块使用门控

只在 TDGX 中使用门控。

TDGX 来自 temporal-difference excitation 思路，因此使用门控注入是合理的：

```text
x'_t = x_t + β · g_t ⊙ v_t
```

其中：

- `g_t` 是 temporal-difference gate。
- `v_t` 是 temporal residual value。
- `β` 是可学习或可调度的残差强度。
- gate 只控制 temporal residual，不直接控制 DINO 主特征。

禁止写成：

```text
x'_t = g_t ⊙ x_t
```

因为这样会直接重标定 DINOv3 feature，容易破坏 baseline 泛化。

### 1.2 哪些模块不使用门控

以下模块不要强行加 gate：

```text
Temporal Tube Dropout
Temporal-Only Attention
Query Volume Decoder
```

它们分别对应 masking、temporal attention、query-based mask decoding，不是 temporal-difference excitation 机制。

------

## 2. 输入输出约定

原始输入：

```python
images: Tensor[B, M, K, 3, 512, 512]
```

其中：

```text
B = batch size
M = num_clips
K = num_frames per clip
```

baseline 配置中通常是：

```text
M = 4
K = 4
```

DINOv3 encoder 输出：

```python
features_flat: Tensor[B*M*K, 1024, 32, 32]
```

在新增视频模块中统一 reshape 成：

```python
G = B * M
x = features_flat.view(G, K, 1024, 32, 32)
```

所有新增视频模块都处理：

```python
x: Tensor[G, K, C, H, W]
```

其中：

```text
C = 1024
H = 32
W = 32
```

最终 decoder 输出：

```python
logits: Tensor[G, K, 1, 512, 512]
logits32: Tensor[G, K, 1, 32, 32]
aux_logits: list[Tensor[G, K, 1, 512, 512]]
```

模型最终返回时可以 reshape 回：

```python
logits: Tensor[B, M, K, 1, 512, 512]
```

------

## 3. 新增模块一：Temporal Tube Dropout

### 3.1 作用

这是训练期正则模块，用来减少模型对固定空间 patch 或固定伪造痕迹的依赖。

它只在训练时启用，验证和测试时必须关闭。

### 3.2 输入输出

```python
input:  x Tensor[G, K, C, H, W]
output: x Tensor[G, K, C, H, W]
```

### 3.3 实现方式

生成空间 tube mask：

```python
mask: Tensor[G, 1, 1, H, W]
```

同一个空间位置 `(h, w)` 在所有时间帧上共享 dropout。

伪代码：

```python
if self.training and drop_prob > 0:
    keep = torch.rand(G, 1, 1, H, W, device=x.device) > drop_prob
    keep = keep.to(x.dtype)
    x = x * keep / (1.0 - drop_prob)
return x
```

注意：

1. 不要按 frame dropout。
2. 不要让整个 frame 消失。
3. 不要在 eval/test 中启用。
4. drop_prob 初始建议 0.10。
5. 如果同源提升不足，可以降到 0.05。
6. 如果跨源泛化掉太多，可以升到 0.15。

### 3.4 文件建议

新增：

```text
src/models/adapters/temporal_tube_dropout.py
```

类名：

```python
class TemporalTubeDropout(nn.Module):
    def __init__(self, drop_prob: float = 0.10):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

------

## 4. 新增模块二：TDGX

TDGX 全称：

```text
Temporal Difference Gated eXcitation
```

这是本方案唯一使用门控注入的模块。

### 4.1 作用

TDGX 在 DINOv3 feature 上计算短期 temporal difference，然后用 gate 控制一个很小的 temporal residual 注入到原 feature 中。

目标是补充视频时序异常 cue，但不让 temporal branch 接管 DINOv3 主干表征。

### 4.2 输入输出

```python
input:  x Tensor[G, K, C, H, W]
output: x Tensor[G, K, C, H, W]
```

### 4.3 推荐结构

```text
x
 ↓
LayerNorm over channel
 ↓
1×1 down projection: C -> hidden_dim
 ↓
temporal differences:
    d1_t = |z_t - z_{t-1}|
    d2_t = |z_t - z_{t-2}|
 ↓
DiffEncoder
 ↓
Gate branch
Value branch
 ↓
x_out = x + β · gate · value
```

推荐先在低维空间做 difference，不要直接在 1024 维上 concat difference，避免显存和参数浪费。

### 4.4 推荐超参

```yaml
tdgx:
  enabled: true
  hidden_dim: 64
  use_adjacent_diff: true
  use_stride2_diff: true
  diff_norm: "ln"

  gate: true
  gate_type: "channel_spatial"
  gate_bias_init: -2.0

  zero_init_value: true
  beta_init: 0.0
  beta_max: 0.03
  drop_path_prob: 0.05
```

如果同源精度仍然不足，可以做激进版：

```yaml
tdgx:
  hidden_dim: 96
  beta_max: 0.05
```

不建议第一版直接使用：

```yaml
hidden_dim: 128
beta_max: 0.10
```

这样过拟合风险会明显增加。

### 4.5 具体实现细节

#### 4.5.1 Channel LayerNorm

输入是 `[G, K, C, H, W]`，LayerNorm 应该作用在 channel 维。

可以实现：

```python
x_perm = x.permute(0, 1, 3, 4, 2)      # [G,K,H,W,C]
x_norm = self.ln(x_perm)
x_norm = x_norm.permute(0, 1, 4, 2, 3) # [G,K,C,H,W]
```

#### 4.5.2 Down projection

对每一帧独立做 1×1 conv：

```python
z = self.down(x_norm.view(G*K, C, H, W))
z = z.view(G, K, hidden_dim, H, W)
```

#### 4.5.3 Temporal difference

```python
d1 = torch.zeros_like(z)
d1[:, 1:] = torch.abs(z[:, 1:] - z[:, :-1])

d2 = torch.zeros_like(z)
d2[:, 2:] = torch.abs(z[:, 2:] - z[:, :-2])
```

第 0 帧和第 1 帧缺失历史帧时直接补零，不要使用未来帧补齐。

#### 4.5.4 DiffEncoder

```python
diff = torch.cat([d1, d2], dim=2)  # [G,K,2*hidden_dim,H,W]
diff = diff.view(G*K, 2*hidden_dim, H, W)
m = self.diff_encoder(diff)
```

推荐 DiffEncoder：

```text
Conv2d(2*hidden_dim, hidden_dim, 1)
GroupNorm
GELU
Depthwise Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
GroupNorm
GELU
Conv2d(hidden_dim, hidden_dim, 1)
```

#### 4.5.5 Gate branch

使用 channel-spatial gate：

```text
channel gate: GAP(m) -> MLP -> C
spatial gate: Conv2d(hidden_dim, 1, 3, padding=1)
gate = sigmoid(channel_gate + spatial_gate)
```

输出：

```python
gate: Tensor[G*K, C, H, W]
```

gate bias 初始化为 `-2.0`，让 gate 初始偏小。

#### 4.5.6 Value branch

```text
Conv2d(hidden_dim, C, 1)
```

最后一层必须 zero-init：

```python
nn.init.zeros_(self.value_proj.weight)
nn.init.zeros_(self.value_proj.bias)
```

#### 4.5.7 β 残差强度

实现一个参数：

```python
self.beta = nn.Parameter(torch.tensor(beta_init))
```

forward 中使用：

```python
beta = torch.clamp(self.beta, 0.0, beta_max)
out = x_flat + beta * gate * value
```

也可以使用 sigmoid 参数化：

```python
beta = beta_max * torch.sigmoid(self.beta_raw)
```

但要保证初始接近 0。

#### 4.5.8 DropPath

TDGX residual 可以加 DropPath：

```python
residual = self.drop_path(gate * value)
out = x + beta * residual
```

drop_path_prob 建议 0.05。

### 4.6 文件建议

新增：

```text
src/models/adapters/tdgx.py
```

类名：

```python
class TemporalDifferenceGatedExcitation(nn.Module):
    def __init__(self, cfg: dict):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

------

## 5. 新增模块三：Temporal-Only Attention

### 5.1 作用

这个模块只在时间维做 attention，不做全局 spatiotemporal attention。

它的目标是让每个空间位置 `(h, w)` 在 K 帧之间交互：

```text
for each spatial location:
    tokens = [x_1(h,w), x_2(h,w), ..., x_K(h,w)]
    apply temporal attention
```

由于 K 通常只有 4，计算量很小。

### 5.2 是否使用门控

不使用门控。

使用 residual + zero-init + alpha_max 控制强度。

### 5.3 输入输出

```python
input:  x Tensor[G, K, C, H, W]
output: x Tensor[G, K, C, H, W]
```

### 5.4 推荐超参

```yaml
temporal_only_attn:
  enabled: true
  dim: 64
  heads: 4
  dropout: 0.0
  use_temporal_pos: true
  zero_init_out: true
  alpha_init: 0.0
  alpha_max: 0.03
  drop_path_prob: 0.10
```

激进版可以试：

```yaml
temporal_only_attn:
  dim: 96
  alpha_max: 0.05
```

不建议第一版用：

```yaml
dim: 128
alpha_max: 0.10
```

### 5.5 实现方式

#### 5.5.1 Down projection

```python
x_flat = x.view(G*K, C, H, W)
z = self.down(x_flat)  # [G*K, D, H, W]
z = z.view(G, K, D, H, W)
```

#### 5.5.2 reshape 成 temporal tokens

```python
z = z.permute(0, 3, 4, 1, 2)      # [G,H,W,K,D]
z = z.reshape(G*H*W, K, D)        # [G*H*W,K,D]
```

#### 5.5.3 temporal positional embedding

```python
z = z + self.temporal_pos[:, :K, :]
```

`temporal_pos` shape：

```python
[1, max_frames, D]
```

#### 5.5.4 Multi-head attention

使用 PyTorch `nn.MultiheadAttention`，建议 `batch_first=True`：

```python
z_attn, _ = self.attn(z, z, z, need_weights=False)
```

#### 5.5.5 Up projection and residual

```python
z_attn = z_attn.reshape(G, H, W, K, D)
z_attn = z_attn.permute(0, 3, 4, 1, 2)  # [G,K,D,H,W]
z_attn = z_attn.reshape(G*K, D, H, W)

res = self.up(z_attn)  # [G*K,C,H,W]
res = res.view(G, K, C, H, W)
```

`self.up` 必须 zero-init。

```python
alpha = torch.clamp(self.alpha, 0.0, alpha_max)
out = x + alpha * self.drop_path(res)
```

### 5.6 文件建议

新增：

```text
src/models/adapters/temporal_only_attention.py
```

类名：

```python
class TemporalOnlyAttention(nn.Module):
    def __init__(self, cfg: dict):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

------

## 6. 新增模块四：Query Volume Decoder

### 6.1 作用

替换原来的 `DINOv3IMLHead`。

原 baseline decoder 是每帧独立 2D segmentation head。Query Volume Decoder 直接对一个 clip 的 `[K, 32, 32]` video volume 做 mask decoding，输出 3D mask volume：

```text
[G, K, 1, 32, 32]
```

它不使用门控。

### 6.2 输入输出

输入：

```python
x: Tensor[G, K, C, H, W]
```

输出：

```python
{
  "logits": Tensor[G, K, 1, 512, 512],
  "logits32": Tensor[G, K, 1, 32, 32],
  "aux_logits": list[Tensor[G, K, 1, 512, 512]],
  "aux_logits32": list[Tensor[G, K, 1, 32, 32]],
  "debug": dict
}
```

### 6.3 推荐超参

```yaml
query_volume_decoder:
  enabled: true
  in_channels: 1024
  query_dim: 256
  num_queries: 8
  num_layers: 3
  num_heads: 8
  ffn_dim: 1024
  dropout: 0.0
  image_size: 512
  use_aux_outputs: true
  combine_method: "soft_or"
```

如果同源仍明显不足，可以试：

```yaml
query_volume_decoder:
  num_queries: 16
  num_layers: 4
```

但第一版建议保持：

```text
num_queries = 8
num_layers = 3
```

### 6.4 Token projection

将 DINO feature 投影到 query_dim：

```python
feat = self.input_proj(x.view(G*K, C, H, W))  # [G*K,D,H,W]
feat = feat.view(G, K, D, H, W)
```

然后 flatten video tokens：

```python
tokens = feat.permute(0, 1, 3, 4, 2)  # [G,K,H,W,D]
tokens = tokens.reshape(G, K*H*W, D)  # [G,N,D]
```

其中：

```text
N = K * H * W
```

baseline 下：

```text
K = 4
H = 32
W = 32
N = 4096
```

### 6.5 Positional encoding

使用 factorized position encoding：

```text
pos = temporal_pos[t] + spatial_pos[h,w]
```

推荐：

```python
temporal_pos: nn.Parameter[1, K_max, D]
spatial_pos: 2D sine-cosine 或 learned [1, H*W, D]
```

为了实现简单，第一版可以使用 learned spatial position：

```python
self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, D))
self.spatial_pos = nn.Parameter(torch.zeros(1, H*W, D))
```

构造：

```python
pos = temporal_pos.repeat_interleave(H*W, dim=1) + spatial_pos.repeat(1, K, 1)
tokens = tokens + pos
```

### 6.6 Query decoder layer

每层包含：

```text
query self-attention
query-to-video cross-attention
FFN
```

推荐顺序：

```text
self_attn -> norm
cross_attn -> norm
ffn -> norm
```

输入 queries：

```python
queries: Tensor[G, Q, D]
```

learned queries：

```python
self.query_embed = nn.Parameter(torch.randn(1, Q, D) * 0.02)
queries = self.query_embed.expand(G, -1, -1)
```

### 6.7 Mask prediction

每层输出 queries 后，都预测 query masks。

做法：

```python
mask_embed = self.mask_embed(queries)  # [G,Q,D]
pixel_embed = self.pixel_embed(feat)   # [G,K,D,H,W]
```

计算：

```python
mask_logits_q = torch.einsum(
    "gqd,gkdhw->gqkhw",
    mask_embed,
    pixel_embed
)
```

shape：

```python
mask_logits_q: Tensor[G, Q, K, H, W]
```

同时预测 query score：

```python
score_logits = self.score_head(queries).squeeze(-1)  # [G,Q]
```

### 6.8 Query 合并方式：soft_or

因为任务是 binary manipulation localization，不需要 Hungarian matching，也不需要类别分类 loss。

推荐用 soft OR 合并 query masks：

```python
score_prob = torch.sigmoid(score_logits)                 # [G,Q]
mask_prob_q = torch.sigmoid(mask_logits_q)               # [G,Q,K,H,W]

weighted_prob_q = mask_prob_q * score_prob[:, :, None, None, None]
prob = 1.0 - torch.prod(1.0 - weighted_prob_q, dim=1)     # [G,K,H,W]

prob = prob.clamp(1e-4, 1.0 - 1e-4)
logits32 = torch.log(prob / (1.0 - prob))
```

然后：

```python
logits32 = logits32.unsqueeze(2)  # [G,K,1,H,W]
```

这种方式比简单平均 query logits 更适合 binary segmentation，因为多个 query 可以共同覆盖不同篡改区域。

### 6.9 Upsample

```python
logits = F.interpolate(
    logits32.view(G*K, 1, H, W),
    size=(512, 512),
    mode="bilinear",
    align_corners=False,
)
logits = logits.view(G, K, 1, 512, 512)
```

### 6.10 Auxiliary outputs

如果 `use_aux_outputs=true`，每个 decoder layer 都输出一个 `aux_logits32` 和 `aux_logits`。

注意：

1. aux outputs 只用于训练。
2. eval/test 时仍然可以返回，但不参与 loss。
3. aux loss 作为一个整体 loss term，不要拆成多个独立 loss 项。

### 6.11 文件建议

新增：

```text
src/models/decoders/query_volume_decoder.py
```

类名：

```python
class QueryVolumeDecoder(nn.Module):
    def __init__(self, cfg: dict):
        ...
    def forward(self, x: torch.Tensor) -> dict:
        ...
```

------

## 7. 新模型类

### 7.1 文件建议

新增：

```text
src/models/b24_dinov3_iml_tdgx_toattn_qvol_video.py
```

类名：

```python
class B24DINOv3IMLTDGXTOAttnQVolVideoModel(nn.Module):
    ...
```

### 7.2 forward 主流程

伪代码：

```python
def forward(self, images, masks=None):
    # images: [B,M,K,3,H,W]
    B, M, K, _, H_img, W_img = images.shape
    G = B * M

    flat_images = images.reshape(B*M*K, 3, H_img, W_img)

    # DINOv3 encoder + LoRA
    features_flat = self.encode_dinov3(flat_images)
    # [B*M*K,1024,32,32]

    C, Hf, Wf = features_flat.shape[1:]
    x = features_flat.view(G, K, C, Hf, Wf)

    if self.training:
        x = self.temporal_tube_dropout(x)

    x = self.tdgx(x)
    x = self.temporal_only_attn(x)

    dec_out = self.query_volume_decoder(x)

    # dec_out logits: [G,K,1,512,512]
    logits = dec_out["logits"].view(B, M, K, 1, 512, 512)
    logits32 = dec_out["logits32"].view(B, M, K, 1, 32, 32)

    out = {
        "logits": logits,
        "logits32": logits32,
        "aux_logits": reshape_aux(dec_out.get("aux_logits", []), B, M, K),
        "aux_logits32": reshape_aux(dec_out.get("aux_logits32", []), B, M, K),
        "debug": dec_out.get("debug", {}),
    }

    return out
```

### 7.3 DINOv3 / LoRA 部分

直接复用现有 B24 模型中的 DINOv3 加载、LoRA 注入、activation checkpoint 逻辑。

不要重新实现 DINOv3 backbone。

------

## 8. builder 和 init 修改

### 8.1 修改 `src/models/__init__.py`

添加：

```python
from .b24_dinov3_iml_tdgx_toattn_qvol_video import B24DINOv3IMLTDGXTOAttnQVolVideoModel
```

### 8.2 修改 `src/models/builder.py`

添加：

```python
if name == "B24DINOv3IMLTDGXTOAttnQVolVideoModel":
    return B24DINOv3IMLTDGXTOAttnQVolVideoModel(cfg)
```

------

## 9. 配置文件模板

新增配置：

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
  name: "B24DINOv3IMLTDGXTOAttnQVolVideoModel"

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
    dropout: 0.05
    layers: "all"
    targets: "attn.qkv"

  temporal_tube_dropout:
    enabled: true
    drop_prob: 0.10
    same_spatial_across_time: true
    apply_on: "dinov3_feature"
    min_keep_frames: 3

  tdgx:
    enabled: true
    hidden_dim: 64
    use_adjacent_diff: true
    use_stride2_diff: true
    diff_norm: "ln"
    gate: true
    gate_type: "channel_spatial"
    gate_bias_init: -2.0
    zero_init_value: true
    beta_init: 0.0
    beta_max: 0.03
    drop_path_prob: 0.05

  temporal_only_attn:
    enabled: true
    dim: 64
    heads: 4
    dropout: 0.0
    use_temporal_pos: true
    zero_init_out: true
    alpha_init: 0.0
    alpha_max: 0.03
    drop_path_prob: 0.10

  query_volume_decoder:
    enabled: true
    in_channels: 1024
    query_dim: 256
    num_queries: 8
    num_layers: 3
    num_heads: 8
    ffn_dim: 1024
    dropout: 0.0
    image_size: 512
    output_resolution: 32
    use_aux_outputs: true
    combine_method: "soft_or"

loss:
  name: "aggressive_video_mask_loss"

  final_mask:
    enabled: true
    bce_weight: 1.0
    dice_weight: 1.0

  aux_mask:
    enabled: true
    weight: 0.40
    bce_weight: 1.0
    dice_weight: 1.0

  edge:
    enabled: true
    weight: 5.0
    width: 7

  temporal_consistency:
    enabled: false
    weight: 0.01
    gamma: 5.0
    warmup_epochs: 30

optimizer:
  learning_rate: 3.0e-4
  lr_lora: 3.0e-4
  lr_tdgx: 1.0e-4
  lr_temporal_attn: 1.0e-4
  lr_decoder: 3.0e-4
  weight_decay: 5.0e-2

scheduler:
  type: "cosine"
  warmup_epochs: 5
  min_lr: 1.0e-6

val_full_video: true
test_full_video: true
test_max_clips: 4

train:
  n_epochs: 500
  save_dir: "runs/b24_dinov3_iml_tdgx_toattn_qvol_tubedrop_video_paperloss_lora32"
  val_interval: 10

validation:
  full_video: true

ddp:
  auto_torchrun: true
  cuda_visible_devices: "0,1"
  nproc_per_node: 2
  dist_backend: "nccl"
  find_unused_parameters: false
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  torchrun_log_dir: "runs/b24_dinov3_iml_tdgx_toattn_qvol_tubedrop_video_paperloss_lora32/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
```

------

## 10. Loss 设计

默认只使用 3 个 loss term。

不要添加 query classification loss。

不要添加 Hungarian matching。

不要添加 optical-flow consistency。

不要添加 frequency loss。

### 10.1 Loss 总式

```text
L = L_final_mask
  + λ_aux  · L_aux_mask
  + λ_edge · L_edge
```

默认：

```text
λ_aux  = 0.40
λ_edge = 5.0
```

最多可选第 4 个：

```text
+ λ_temp · L_temporal_consistency
```

但默认关闭。

------

## 11. Loss 1：Final Mask Loss

### 11.1 作用

监督最终输出 mask。

### 11.2 输入

```python
pred_logits: Tensor[B, M, K, 1, 512, 512]
gt_masks:    Tensor[B, M, K, 1, 512, 512]
```

### 11.3 组成

Final Mask Loss 是一个 composite loss，但在日志中作为一个主 loss term：

```text
L_final_mask = BCEWithLogits + SoftDice
```

推荐实现：

```python
bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks.float())
dice = soft_dice_loss(pred_logits, gt_masks)
loss = bce_weight * bce + dice_weight * dice
```

默认：

```yaml
bce_weight: 1.0
dice_weight: 1.0
```

------

## 12. Loss 2：Aux Mask Loss

### 12.1 作用

监督 Query Volume Decoder 中间层输出，加速收敛，提高 decoder 稳定性。

### 12.2 输入

```python
aux_logits: list[Tensor[B, M, K, 1, 512, 512]]
gt_masks: Tensor[B, M, K, 1, 512, 512]
```

### 12.3 实现

对每个 aux output 计算同样的 BCE + SoftDice，然后平均：

```python
loss_aux = 0
for aux in aux_logits:
    loss_aux += BCE(aux, gt) + Dice(aux, gt)
loss_aux = loss_aux / max(len(aux_logits), 1)
```

总权重：

```yaml
aux_mask:
  weight: 0.40
```

如果训练不稳定或过拟合，可以降到：

```yaml
weight: 0.20
```

如果 QueryVolumeDecoder 收敛慢，可以升到：

```yaml
weight: 0.60
```

------

## 13. Loss 3：Edge Loss

### 13.1 作用

提升边界、小篡改区域和细碎区域的同源精度。

### 13.2 实现方式

从 GT mask 中提取 edge band。

推荐做法：

1. 对 GT mask 做 max_pool 得到 dilation。
2. 对 `1 - GT` 做 max_pool 得到 erosion 的反向。
3. edge band = dilation - erosion。
4. 在 edge band 上计算 BCEWithLogits。

伪代码：

```python
def mask_to_edge_band(mask, width=7):
    pad = width // 2
    dilated = F.max_pool2d(mask, kernel_size=width, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=width, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0, 1)
    return edge
```

对于视频维度，先 flatten：

```python
pred = pred_logits.view(B*M*K, 1, 512, 512)
gt = gt_masks.view(B*M*K, 1, 512, 512)
edge = mask_to_edge_band(gt, width=7)
```

edge loss：

```python
bce_map = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
loss_edge = (bce_map * edge).sum() / edge.sum().clamp_min(1.0)
```

默认：

```yaml
edge:
  enabled: true
  weight: 5.0
  width: 7
```

如果边界过拟合或跨域下降明显，可以降到：

```yaml
weight: 2.0
```

如果同源边界仍明显不足，可以升到：

```yaml
weight: 10.0
```

不建议第一版使用 20.0。

------

## 14. Optional Loss 4：Feature-aware Temporal Consistency

默认关闭。

只在 full-video 预测明显闪烁、相邻帧 mask 不稳定时启用。

### 14.1 约束

不使用光流。

不做 warp。

只用 DINO feature similarity 控制 temporal consistency 权重。

### 14.2 形式

```text
L_temp = mean( exp(-γ · feature_diff) · |p_t - p_{t-1}| )
```

其中：

```text
p_t = sigmoid(logits_t)
feature_diff = mean_channel(|norm_feat_t - norm_feat_{t-1}|)
```

feature_diff 大的区域，说明视觉变化大，降低一致性约束。

feature_diff 小的区域，说明视觉稳定，鼓励预测稳定。

### 14.3 默认配置

```yaml
temporal_consistency:
  enabled: false
  weight: 0.01
  gamma: 5.0
  warmup_epochs: 30
```

启用时必须在 30 epoch 后再打开，避免早期抑制模型学习。

------

## 15. Loss 文件建议

新增：

```text
src/losses/aggressive_video_mask_loss.py
```

类名：

```python
class AggressiveVideoMaskLoss(nn.Module):
    def __init__(self, cfg: dict):
        ...
    def forward(self, outputs: dict, targets: torch.Tensor, extra: dict | None = None) -> dict:
        ...
```

返回格式建议：

```python
{
  "loss": total_loss,
  "loss_final_mask": loss_final.detach(),
  "loss_aux_mask": loss_aux.detach(),
  "loss_edge": loss_edge.detach(),
  "loss_temp": loss_temp.detach() if enabled else 0,
}
```

修改 loss builder，使：

```yaml
loss:
  name: "aggressive_video_mask_loss"
```

可以构建该 loss。

------

## 16. Optimizer 参数组

需要给新增模块不同学习率。

建议参数组：

```text
LoRA:                lr_lora = 3e-4
TDGX:                lr_tdgx = 1e-4
TemporalOnlyAttn:    lr_temporal_attn = 1e-4
QueryVolumeDecoder:  lr_decoder = 3e-4
```

DINOv3 backbone 继续 frozen。

不要全量 fine-tune DINOv3。

### 16.1 参数命名建议

确保模块名清晰，便于 optimizer 分组：

```python
self.tdgx
self.temporal_only_attn
self.query_volume_decoder
```

optimizer builder 中按名字匹配：

```python
if "tdgx" in name:
    lr = lr_tdgx
elif "temporal_only_attn" in name:
    lr = lr_temporal_attn
elif "query_volume_decoder" in name:
    lr = lr_decoder
elif "lora" in name:
    lr = lr_lora
```

------

## 17. 初始化要求

### 17.1 TDGX

必须满足：

```text
value branch final conv zero-init
beta 初始接近 0
gate bias 初始化为 -2.0
```

### 17.2 TemporalOnlyAttention

必须满足：

```text
up projection zero-init
alpha 初始接近 0
```

### 17.3 QueryVolumeDecoder

推荐：

```text
query embedding normal std=0.02
input projection kaiming/xavier
cross-attn/self-attn 使用 PyTorch 默认或 xavier
mask_embed 最后一层不要 zero-init
score_head bias 初始化为 0
```

不要把 QueryVolumeDecoder 整体 zero-init，否则训练初期可能几乎没有 mask 学习信号。

------

## 18. 训练建议

### 18.1 第一阶段：基础稳定版

先跑完整 final config，但保持：

```yaml
tdgx:
  hidden_dim: 64
  beta_max: 0.03

temporal_only_attn:
  dim: 64
  alpha_max: 0.03

query_volume_decoder:
  num_queries: 8
  num_layers: 3

temporal_tube_dropout:
  drop_prob: 0.10

edge:
  weight: 5.0
```

### 18.2 如果同源精度不够

按顺序调：

```text
1. query_volume_decoder.num_queries: 8 -> 16
2. query_volume_decoder.num_layers: 3 -> 4
3. tdgx.hidden_dim: 64 -> 96
4. temporal_only_attn.dim: 64 -> 96
5. edge.weight: 5.0 -> 10.0
```

一次只改一个。

### 18.3 如果跨域泛化掉太多

按顺序调：

```text
1. temporal_tube_dropout.drop_prob: 0.10 -> 0.15
2. edge.weight: 5.0 -> 2.0
3. tdgx.beta_max: 0.03 -> 0.02
4. temporal_only_attn.alpha_max: 0.03 -> 0.02
5. lora.dropout: 0.05 -> 0.10
```

一次只改一个。

------

## 19. Ablation 顺序

必须做 ablation，否则无法判断模块贡献。

建议顺序：

```text
A0: 原 b24 baseline
A1: baseline + QueryVolumeDecoder
A2: A1 + TDGX
A3: A2 + TemporalOnlyAttention
A4: A3 + TubeDrop
A5: A4 + Edge Loss
```

如果时间有限，至少做：

```text
A0: baseline
A1: baseline + QueryVolumeDecoder
A2: baseline + QueryVolumeDecoder + TDGX
A3: full model
```

其中 full model：

```text
baseline + TDGX + TemporalOnlyAttention + QueryVolumeDecoder + TubeDrop + 3-loss setup
```

------

## 20. 验收标准

### 20.1 Shape test

必须写最小单元测试：

```python
B, M, K = 1, 4, 4
x = torch.randn(B, M, K, 3, 512, 512).cuda()
out = model(x)

assert out["logits"].shape == (B, M, K, 1, 512, 512)
assert out["logits32"].shape == (B, M, K, 1, 32, 32)
```

### 20.2 TDGX identity test

当：

```text
beta = 0
```

TDGX 输出必须与输入几乎一致：

```python
max_abs_diff < 1e-6
```

### 20.3 TemporalOnlyAttention identity test

当：

```text
alpha = 0
```

TemporalOnlyAttention 输出必须与输入几乎一致：

```python
max_abs_diff < 1e-6
```

### 20.4 QueryVolumeDecoder numerical test

必须检查：

```text
logits 无 NaN
logits 无 Inf
prob clamp 生效
soft_or 输出范围稳定
```

### 20.5 Loss test

给随机 logits 和随机 binary mask，loss 必须：

```text
可 forward
可 backward
无 NaN
loss > 0
```

------

## 21. 评估要求

必须分别报告：

```text
DVI validation
CPNET validation
DVI + CPNET validation average
OPN test
full-video validation
clip-level validation
```

不要只报平均值。

必须记录：

```text
IoU
F1
Precision
Recall
AUC if available
Boundary F1 if available
```

如果只选一个主指标，使用 IoU 或 F1。

### 21.1 threshold

默认先使用固定阈值：

```text
threshold = 0.5
```

可以额外报告 val-best threshold，但不能用 test set 调阈值。

------

## 22. 不允许做的事

本方案禁止：

```text
1. 添加 RAFT / PWCNet / GMFlow / FlowFormer 等光流网络
2. 添加频域 FFT / DCT / wavelet 分支
3. 添加音频输入
4. 添加第二个大型 image/video backbone
5. 全量 fine-tune DINOv3
6. 添加 Hungarian matching
7. 添加 query classification loss
8. 添加超过 4 个 loss term
9. 把所有模块都改成 gate
10. 使用全局 K*H*W token self-attention
```

TemporalOnlyAttention 只能在每个空间位置沿时间维做 attention。

------

## 23. 推荐最终默认版本

最终默认版本应为：

```text
B24DINOv3IMLTDGXTOAttnQVolVideoModel
```

默认模块：

```text
TDGX: enabled
TemporalOnlyAttention: enabled
QueryVolumeDecoder: enabled
TemporalTubeDropout: enabled
```

默认 loss：

```text
1. Final Mask Loss = BCEWithLogits + SoftDice
2. Aux Mask Loss = BCEWithLogits + SoftDice
3. Edge Loss = edge-band BCE
```

默认不开：

```text
Temporal Consistency Loss
```

如果 full-video mask 抖动明显，再启用第 4 个 loss。

------

## 24. 最小实现清单

需要新增或修改以下文件：

```text
src/models/adapters/temporal_tube_dropout.py
src/models/adapters/tdgx.py
src/models/adapters/temporal_only_attention.py
src/models/decoders/query_volume_decoder.py
src/models/b24_dinov3_iml_tdgx_toattn_qvol_video.py
src/models/__init__.py
src/models/builder.py
src/losses/aggressive_video_mask_loss.py
src/losses/builder.py
configs/b24_dinov3_iml_tdgx_toattn_qvol_tubedrop_video_paperloss_lora32.yml
```

如果当前项目没有 `src/losses/builder.py`，则在现有 loss 构建位置加入：

```python
if loss_name == "aggressive_video_mask_loss":
    return AggressiveVideoMaskLoss(loss_cfg)
```

------

## 25. 实现优先级

按以下顺序实现：

```text
1. QueryVolumeDecoder
2. 新模型类接入 QueryVolumeDecoder
3. AggressiveVideoMaskLoss
4. TDGX
5. TemporalOnlyAttention
6. TemporalTubeDropout
7. config
8. builder / __init__ 接入
9. shape test / identity test / loss test
10. 跑 ablation
```

不要一开始同时 debug 所有模块。

建议先保证：

```text
baseline + QueryVolumeDecoder
```

能正常训练，再加入 TDGX 和 TemporalOnlyAttention。

------

## 26. 预期效果

该方案相比 baseline 预期：

```text
同源 DVI/CPNET: 明显提升
边界和小区域: 明显提升
clip 内一致性: 提升
跨源 OPN: 可能略降，但应通过 TubeDrop、LoRA dropout、低 beta/alpha 控制
```

如果 OPN 明显下降，优先降低 temporal 模块强度，而不是删除 QueryVolumeDecoder。

优先调整：

```text
tdgx.beta_max
temporal_only_attn.alpha_max
temporal_tube_dropout.drop_prob
edge.weight
lora.dropout
```

------

## 27. 最终结论

本方案的核心是：

```text
用 QueryVolumeDecoder 提升 mask 表达上限；
用 TDGX 注入论文合理的 temporal-difference gated cue；
用 TemporalOnlyAttention 补充短时 token interaction；
用 TubeDrop 和低强度 residual 控制泛化风险；
loss 保持 3 个必要项，不做复杂 loss 堆叠。
```

这是一个 accuracy-oriented 的激进方案，但仍然遵守：

```text
无光流
无频域
无音频
无大额外 backbone
少 loss
尊重视频论文原始机制
```