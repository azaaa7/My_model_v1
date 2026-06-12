# B23-TTF-CCM-Lite-LoRA32 实现任务书：DINOv3 Patch Embedding 后时序 Token Fusion + 非顺序依赖数据流 + 结构化 Loss 重构

> 目标：在当前 `b23_ccm_lite_lora32.yml` 的强泛化 baseline 上，新增 **DINOv3 patch embedding 后、Transformer blocks 前的 Temporal Token Fusion（TTF）**。  
> 该方案不依赖 FGM 的 full-video stateful bank，不要求训练/验证/测试按整视频顺序输入。  
> 同时重构 loss，从“多个 loss 简单堆叠”改成面向视频 inpainting 检测的 **区域-边界-时序-Token 结构化监督**。

---

## 0. 给 coding agent 的总要求

请在一个实现分支中完成以下目标：

1. 新增 `TemporalPatchTokenFusion` 模块。
2. 新增 `DINOv3B23TemporalEncoder`，在 DINOv3 `patch_embed` 后、DINO blocks 前注入跨帧 token 融合。
3. 修改主模型 `B23TFCUCCMFGMLiteModel`，让它支持：
   - 原始 encoder：`DINOv3B23Encoder`
   - 新 temporal encoder：`DINOv3B23TemporalEncoder`
4. 设计新的配置：
   - `configs/b23_ttf_ccm_structloss_lora32.yml`
5. 训练/验证/测试输入流程改成 **无 FGM 顺序依赖**：
   - 不需要 `stateful_train`
   - 不需要 `stateful_eval`
   - 不需要同一视频内部 window 按顺序喂给模型
   - 只保留 clip/window 内的短程多帧关系
6. 新增结构化 loss：
   - 主区域监督：balanced focal BCE + focal Tversky / Dice
   - 边界监督：boundary band loss
   - 多尺度深监督：mask128 / mask256 / ccm_mask32，但要归一化和分组，而不是无脑相加
   - 顺序无关的跨帧一致性：pairwise temporal delta loss
   - 可选 token 级 anomaly separation loss
   - 可选 TTF residual regularization
7. 保证所有新增模块都是 **zero-init residual** 或初始近似 identity。
8. 保证当 `temporal_encoder.enabled=false` 时，旧配置和旧结果路径不受影响。
9. 完成 shape check、禁用态等价测试、loss 数值稳定测试和 ablation 配置。

---

## 1. 当前 baseline 诊断

当前 `b23_ccm_lite_lora32.yml` 的核心流程是：

```text
video: [B, M, K, 3, 512, 512]
        |
        | reshape
        v
frames: [B*M*K, 3, 512, 512]
        |
        | DINOv3B23Encoder
        v
b23: [B*M*K, 1024, 32, 32]
        |
        | reshape back
        v
features: [B, M, K, 1024, 32, 32]
        |
        | clip-wise CCM-Lite
        v
f_cc: [B, K, 1024, 32, 32]
        |
        | LowResTFCUFusion
        v
f32: [B*K, 128, 32, 32]
        |
        | LiteBoundaryDecoder
        v
logits: [B, M, K, 1, 512, 512]
```

当前问题：

1. DINOv3 实际上只看到单帧 batch，不知道视频维度。
2. CCM 是 DINOv3 后处理时序模块，无法让 DINOv3 block 内部的强空间交互提前感知跨帧异常。
3. FGM 通过历史 bank 做跨 window 传播，容易依赖视频顺序和源域记忆。
4. 当前 loss 主要是 Dice/BCE/Tversky/Boundary/Aux 叠加，缺少“哪些 loss 负责区域、哪些负责边界、哪些负责时序、哪些约束 token”的清晰分工。
5. 目前已有 `temporal_delta`，但它按相邻帧顺序做差，对“无顺序依赖、多 clip 随机输入”的新目标不够合适。

---

## 2. 新方案核心：B23-TTF-CCM-Lite-LoRA32

新结构：

```text
video: [B, M, K, 3, 512, 512]
        |
        | normalize
        v
frames: [B*M*K, 3, 512, 512]
        |
        | DINOv3 patch_embed + cls/storage tokens
        v
patch tokens: [B*M*K, N, C]
N = 32*32 = 1024, C = 1024
        |
        | reshape to video token layout
        v
tokens: [B, M, K, N, C]
        |
        | TemporalPatchTokenFusion
        | same spatial patch, cross-frame K attention
        v
tokens': [B, M, K, N, C]
        |
        | reshape back
        v
patch tokens': [B*M*K, N, C]
        |
        | concat cls/storage tokens
        v
DINOv3 blocks 0..23
        |
        | take block 23 patch tokens
        v
b23: [B*M*K, 1024, 32, 32]
        |
        | original CCM-Lite + decoder
        v
logits: [B, M, K, 1, 512, 512]
```

关键设计原则：

- **不改 RGB 输入通道**，不把多帧拼成 9/12 通道。
- **不把多帧融合成一张图**，避免破坏 DINOv3 的自然图像输入分布。
- 只在 patch token 层做轻量时序融合。
- 融合只发生在同一空间 patch 位置的 K 帧之间。
- DINOv3 后续 block 继续负责强空间交互。
- TTF 使用 zero-init residual，初始等价原模型。
- 不使用 full-video memory bank，不需要 window 顺序。

---

## 3. 新增文件结构

请新增或修改以下文件：

```text
src/models/
├── dinov3_b23_encoder.py                    # 保留原始 encoder
├── dinov3_b23_temporal_encoder.py           # 新增
├── b23_tfcu_ccm_fgm_model.py                # 修改：支持 temporal encoder
├── tfcu/
│   ├── temporal_token_fusion.py             # 新增
│   ├── ccm_lite.py
│   ├── fgm_lite.py
│   └── fusion.py
├── decoders/
│   └── lite_boundary_decoder.py
└── lora.py

src/losses/
├── segmentation_losses.py                   # 可保留 legacy
├── structured_forensic_loss.py              # 新增，推荐
├── aux_losses.py                            # 可保留 legacy
├── boundary_loss.py
└── __init__.py                              # 导出新 loss

src/train/
├── trainer.py                               # 修改：支持 CompositeForensicLoss
├── optimizer.py                             # 修改：新增 temporal encoder 参数组
└── scheduler.py

configs/
├── b23_ccm_lite_lora32.yml                  # 不动
└── b23_ttf_ccm_structloss_lora32.yml        # 新增
```

---

## 4. TemporalPatchTokenFusion 设计

### 4.1 输入输出

```python
Input:
    x: [B, M, K, N, C]

Output:
    x_out: [B, M, K, N, C]
    debug: dict
```

其中：

```text
B = batch size
M = num_clips
K = num_frames per clip
N = patch count = 32 * 32
C = DINOv3 feature dim = 1024
```

### 4.2 融合方式

仅在同一 patch 位置跨 K 帧做 temporal attention：

```text
for each b, m, n:
    sequence = x[b, m, :, n, :]    # [K, C]
    sequence' = TemporalAttention(sequence)
```

不做全空间 attention，不做跨 clip memory。

### 4.3 模块结构

```text
x
 |
 | permute/reshape
 v
[B*M*N, K, C]
 |
 | LayerNorm
 | Linear C -> bottleneck_dim
 | temporal Q/K/V attention over K
 | Linear bottleneck_dim -> C
 | zero-init projection
 v
residual
 |
 | x + alpha * residual
 v
x_out
```

推荐参数：

```yaml
temporal_encoder:
  enabled: true
  type: "patch_token_fusion"
  insert_position: "after_patch_embed_before_blocks"

  token_fusion:
    enabled: true
    mode: "same_patch_temporal_attention"
    dim: 1024
    bottleneck_dim: 128
    heads: 4
    dropout: 0.05
    frame_mask: "bidirectional"
    alpha_init: 0.0
    proj_zero_init: true
    return_residual_energy: true
```

---

## 5. DINOv3B23TemporalEncoder 设计

### 5.1 为什么要新增 encoder

当前 `DINOv3B23Encoder.forward(frames)` 直接调用：

```python
self.backbone.get_intermediate_layers(frames, n=[self.output_block], reshape=True, norm=True)
```

这个黑盒接口无法在 patch embedding 后插入 TTF。  
因此需要新增 encoder，手动展开 DINOv3 forward：

```text
normalize
patch_embed / prepare_tokens_with_masks
split extra tokens and patch tokens
TTF
concat tokens
run blocks 0..output_block
norm
reshape patch tokens to [N, C, H, W]
```

### 5.2 文件

创建：

```text
src/models/dinov3_b23_temporal_encoder.py
```

### 5.3 实现要点

1. 复用 `load_dinov3_backbone()`。
2. 复用现有 LoRA 注入逻辑。
3. 支持 `freeze_backbone=true`。
4. TTF 参数必须 trainable。
5. 手写 forward 时要兼容 DINOv3 local repo 的具体 API：
   - 优先使用 `prepare_tokens_with_masks(frames)`
   - 如果返回值不是 `(x, (ph, pw))`，agent 需要根据本地 DINOv3 代码适配。
6. 输出必须严格为：

```python
[B*M*K, 1024, 32, 32]
```

7. debug 中至少包含：
   - `temporal_encoder_enabled`
   - `ttf_alpha`
   - `patch_hw`
   - `feat_shape`

---

## 6. 修改主模型

文件：

```text
src/models/b23_tfcu_ccm_fgm_model.py
```

### 6.1 初始化逻辑

当前：

```python
self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
```

改成：

```python
temporal_encoder_cfg = cfg.get("temporal_encoder", {}) or {}
self.use_temporal_encoder = bool(temporal_encoder_cfg.get("enabled", False))

if self.use_temporal_encoder:
    from .dinov3_b23_temporal_encoder import DINOv3B23TemporalEncoder
    self.encoder = DINOv3B23TemporalEncoder(
        dinov3_cfg=dinov3_cfg,
        lora_cfg=lora_cfg,
        temporal_cfg=temporal_encoder_cfg,
    )
else:
    self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
```

### 6.2 新增 encode_video_or_frames

替代只接受 frames 的 `encode_frames`：

```python
def encode_video_or_frames(self, video: torch.Tensor) -> tuple[torch.Tensor, dict]:
    b, m, k, c, h, w = video.shape

    if self.use_temporal_encoder:
        feat, debug = self.encoder(video)
        return feat, debug

    frames = video.reshape(b * m * k, c, h, w)
    feat = self.encode_frames(frames)
    return feat, {"temporal_encoder_enabled": False}
```

### 6.3 forward 中替换

当前：

```python
frames = video.reshape(b * m * k, c, h, w)
b23 = self.encode_frames(frames)
```

改为：

```python
frames = video.reshape(b * m * k, c, h, w)
b23, temporal_debug = self.encode_video_or_frames(video)
```

debug 里加入：

```python
debug = {
    "input_video_shape": tuple(video.shape),
    "b23_feature_shape": tuple(b23.shape),
    "temporal_encoder": temporal_debug,
    "forensic_branch": forensic_debug,
}
```

### 6.4 FGM 默认关闭

此方案不删除 FGM 代码，但新配置必须：

```yaml
tfcu:
  fgm:
    enabled: false

fgm_bank:
  train_full_video_windows: false
  stateful_train: false
  stateful_eval: false
```

---

## 7. 数据训练/验证/测试输入流程重设

### 7.1 原则

新 TTF 只依赖当前输入 window 内的 `[M,K]` 多帧 token，不依赖历史 bank。  
因此训练、验证、测试都不要求按整视频顺序输入。

### 7.2 训练输入

推荐：

```text
每个 sample 随机选一个视频
随机采样 M 个 clip
每个 clip 内 K 帧
输出 [M,K,3,512,512] 和 [M,K,1,512,512]
DataLoader 可以 shuffle
DDP 可以普通 DistributedSampler
```

配置：

```yaml
num_clips: 4
num_frames: 4
clip_stride: 1

fgm_bank:
  train_full_video_windows: false
  stateful_train: false
  stateful_eval: false

val_full_video: false
test_full_video: false
```

保留轻量 temporal augment：

```yaml
temporal_augment:
  frame_swap:
    enabled: true
    prob: 0.10
    max_swaps: 1
    local_radius: 2
  frame_drop:
    enabled: true
    prob: 0.10
    max_drops: 1
```

### 7.3 验证输入

不要求 full-video stateful eval。推荐：

```text
val: 对每个视频 deterministic 采样 M 个 clip
模型一次处理 [1,M,K,3,512,512]
直接算 frame-level mask metrics
```

配置：

```yaml
val_full_video: false
```

如果希望覆盖整视频，也可以使用 full-video windows，但不要启用 stateful bank：

```yaml
val_full_video: true
fgm_bank:
  stateful_eval: false
```

此时 window 顺序只影响输出日志顺序，不影响模型结果。

### 7.4 测试输入

两种模式均可：

**快速测试：**

```yaml
test_full_video: false
test_max_clips: 4
```

**整视频覆盖测试：**

```yaml
test_full_video: true
test_max_clips: 4
fgm_bank:
  stateful_eval: false
```

整视频覆盖模式下，dataset 仍可产生 sequential windows，但模型不携带跨 window 状态。因此：

- 可以按顺序跑
- 也可以被 DDP 切分
- 不需要 `VideoWindowSampler` 保证同视频 window 连续
- 不需要 `reset_on_new_video`

---

## 8. Loss 重构：从堆叠到结构化目标

### 8.1 当前 loss 问题

旧配置大致是：

```yaml
loss:
  dice: {weight: 1.0}
  bce: {weight: 0.5}
  tversky: {weight: 0.2}
  boundary: {weight: 0.15}

aux_loss:
  ccm_mask32: {enabled: true, weight: 0.05}
  mask128: {enabled: true, weight: 0.20}
  boundary128: {enabled: true, weight: 0.10}
```

问题不是这些 loss 错，而是它们职责重叠：

- Dice/Tversky 都在处理区域重叠。
- BCE 和 Dice 同时管像素分类，但类别不平衡没有被显式建模。
- Boundary 和 boundary128 是两个边界项，但没有 boundary band 的空间重点。
- Aux loss 是简单加权，没有 group normalization。
- 时序 loss 没有和“无顺序依赖”的新输入方式匹配。
- TTF token 的时序异常提示没有任何直接约束，可能学成噪声或源域 shortcut。

### 8.2 新 loss 总体形式

新 loss 建议命名：

```python
CompositeForensicLoss
```

总目标：

```text
L_total =
    w_region   * L_region
  + w_boundary * L_boundary
  + w_temporal * ramp(epoch) * L_pairwise_temporal
  + w_deep     * L_deep_supervision
  + w_token    * ramp(epoch) * L_token_separation
  + w_reg      * ramp(epoch) * L_ttf_regularization
```

其中：

```text
L_region:
    balanced focal BCE + focal Tversky

L_boundary:
    boundary band BCE/Dice

L_pairwise_temporal:
    order-invariant pairwise delta consistency

L_deep_supervision:
    mask128 / mask256 / ccm_mask32 分组归一化

L_token_separation:
    32x32 token inside/outside anomaly separation，可选

L_ttf_regularization:
    TTF residual energy 不要在 GT 无变化区域过强，可选
```

### 8.3 新配置

```yaml
loss:
  type: "composite_forensic"

  region:
    enabled: true
    weight: 1.0
    focal_bce:
      weight: 0.6
      gamma: 2.0
      alpha: 0.75
      adaptive_pos_weight: true
      pos_weight_clip: [1.0, 20.0]
    focal_tversky:
      weight: 0.4
      alpha: 0.3
      beta: 0.7
      gamma: 0.75
      smooth: 1.0e-6

  boundary:
    enabled: true
    weight: 0.20
    kernel_size: 5
    band_width: 5
    bce_weight: 0.7
    dice_weight: 0.3

  pairwise_temporal:
    enabled: true
    weight: 0.08
    warmup_epochs: 20
    max_pairs: 24
    downsample: 128
    positive_delta_weight: 1.0
    negative_delta_weight: 0.35
    loss_type: "charbonnier"

  deep_supervision:
    enabled: true
    weight: 0.25
    normalize_by_active_heads: true
    heads:
      mask128: {enabled: true, weight: 0.50}
      mask256: {enabled: true, weight: 0.25}
      ccm_mask32: {enabled: true, weight: 0.25}
      fgm_mask32: {enabled: false, weight: 0.0}
      boundary128: {enabled: true, weight: 0.25}

  token_separation:
    enabled: false
    weight: 0.03
    warmup_epochs: 50
    min_pixels_per_class: 8
    max_tokens_per_frame: 256
    margin: 0.20

  ttf_regularization:
    enabled: false
    weight: 0.005
    warmup_epochs: 20
    nochange_weight: 1.0
```

第一轮建议：

```yaml
token_separation.enabled: false
ttf_regularization.enabled: false
```

先验证 TTF + pairwise temporal loss 是否稳定。稳定后再打开 token 级 loss。

---

## 9. 新 Loss 具体设计

### 9.1 RegionLoss：Balanced Focal BCE + Focal Tversky

#### 9.1.1 Balanced Focal BCE

目的：处理 mask 前景稀疏，避免模型只学背景。

公式：

```text
p = sigmoid(logits)
BCE = binary_cross_entropy_with_logits(logits, target)

pt = p       if target = 1
pt = 1 - p   if target = 0

focal = alpha_t * (1 - pt)^gamma * BCE
```

`adaptive_pos_weight`：

```text
pos = target.sum()
neg = target.numel() - pos
pos_weight = clamp(neg / max(pos, eps), low, high)
```

注意：  
如果使用 `pos_weight`，不要再过度增大 `alpha`，否则前景会过拟合。推荐：

```text
gamma=2.0
alpha=0.75
pos_weight_clip=[1,20]
```

#### 9.1.2 Focal Tversky

```text
Tversky = TP / (TP + alpha * FP + beta * FN)
L = (1 - Tversky)^gamma
```

推荐：

```text
alpha=0.3
beta=0.7
gamma=0.75
```

因为 inpainting 检测更怕漏检，`beta` 可以大于 `alpha`。

---

### 9.2 BoundaryBandLoss

当前 boundary loss 用 Sobel L1。建议改成 boundary band 监督：

1. 从 GT mask 生成 edge：

```text
edge = dilate(mask) - erode(mask)
```

2. 生成 boundary band：

```text
band = dilate(edge, band_width)
```

3. 在 band 区域上计算 BCE 和 Dice：

```text
L_boundary =
    bce_weight  * BCE(pred * band, edge)
  + dice_weight * Dice(pred_edge, edge)
```

目的：

- 边界区域权重大。
- 非边界区域不让 boundary loss 过度惩罚。
- 更适合 inpainting 区域边缘定位。

---

### 9.3 PairwiseTemporalDeltaLoss：顺序无关时序监督

旧 `temporal_delta` 是相邻帧：

```text
|p_t - p_{t-1}| vs |y_t - y_{t-1}|
```

新方案不依赖顺序，因此使用所有帧对或采样帧对：

```text
for all pairs i < j in current input window:
    pred_delta = |sigmoid(logit_i) - sigmoid(logit_j)|
    gt_delta   = |mask_i - mask_j|
    loss += rho(pred_delta - gt_delta)
```

其中 `rho` 推荐 Charbonnier：

```text
rho(x) = sqrt(x^2 + eps^2)
```

输入：

```text
logits: [B,M,K,1,H,W]
target: [B,M,K,1,H,W]
```

先 flatten：

```text
T = M*K
logits_t: [B,T,1,H,W]
target_t: [B,T,1,H,W]
```

为了省显存，可以 downsample 到 128：

```text
logits_t -> [B,T,1,128,128]
target_t -> [B,T,1,128,128]
```

正负权重：

```text
gt_delta > 0:
    positive_delta_weight = 1.0

gt_delta == 0:
    negative_delta_weight = 0.35
```

这样可以防止模型在 GT 没有变化的区域产生乱跳，同时不过度强制所有帧完全一致。

---

### 9.4 DeepSupervisionLoss：分组归一化

旧 aux loss 是直接：

```text
0.05 * ccm_mask32 + 0.20 * mask128 + 0.10 * boundary128
```

建议改成 group：

```text
L_deep =
    sum(active_head_weight * head_loss) / sum(active_head_weight)
```

然后再乘总权重：

```text
L_total += deep_supervision.weight * L_deep
```

每个 head 使用：

```text
mask head:
    focal BCE + dice

boundary head:
    boundary band BCE
```

不要让 deep supervision 的实际总权重随打开 head 数量变化太大。

---

### 9.5 TokenSeparationLoss，可选

TTF 是 token 层时序融合模块，最终输出 32x32 patch token。可以让 agent 预留接口：

```python
aux["ttf_tokens"] = patch_after_ttf.detach_or_not
aux["ttf_residual_energy"] = residual_energy
```

用 GT mask 下采样到 32x32：

```text
gt32: [B,M,K,1,32,32]
```

定义前景 token 和背景 token：

```text
fg_tokens = tokens where gt32 > 0.5
bg_tokens = tokens where gt32 < 0.1
```

做 prototype margin loss：

```text
fg_proto = mean(normalize(fg_tokens))
bg_proto = mean(normalize(bg_tokens))

L_token = max(0, cosine(fg_proto, bg_proto) + margin)
```

也可以采样 token 做 supervised contrastive。第一版为了稳定，建议先关闭：

```yaml
token_separation.enabled: false
```

---

### 9.6 TTFRegularization，可选

防止 TTF 在 GT 无变化区域乱注入：

```text
pair_gt_delta = |y_i - y_j|
nochange = pair_gt_delta < threshold

L_reg = mean(ttf_energy * nochange)
```

或更简单：

```text
L_reg = mean(ttf_residual_energy on background/no-change tokens)
```

第一版先关闭，等 TTF 学稳再开。

---

## 10. structured_forensic_loss.py 实现骨架

创建：

```text
src/losses/structured_forensic_loss.py
```

建议结构：

```python
class CompositeForensicLoss(nn.Module):
    def __init__(self, cfg):
        ...

    def forward(self, logits, target, aux=None, epoch: int | None = None):
        total = 0
        items = {}

        region = self.region_loss(logits, target)
        boundary = self.boundary_loss(logits, target)
        temporal = self.pairwise_temporal_loss(logits, target)
        deep = self.deep_supervision_loss(aux, target)
        token = self.token_separation_loss(aux, target)
        reg = self.ttf_regularization(aux, target)

        total = weighted sum with ramp

        return total, items
```

### 10.1 ramp 函数

```python
def ramp_weight(base_weight: float, epoch: int | None, warmup_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if epoch is None or warmup_epochs <= 0:
        return base_weight
    ratio = min(1.0, max(0.0, epoch / float(warmup_epochs)))
    return base_weight * ratio
```

### 10.2 兼容 trainer

当前 trainer 是：

```python
main_loss, main_items = criterion(logits, target)
aux_loss, aux_items = aux_criterion(aux, target)
total_loss = main_loss + aux_loss
```

新逻辑建议改成：

```python
if isinstance(criterion, CompositeForensicLoss):
    total_loss, loss_items = criterion(logits, target, aux=out["aux"], epoch=epoch)
    main_loss = total_loss
    aux_loss = total_loss * 0.0
    main_items = loss_items
    aux_items = {}
else:
    main_loss, main_items = criterion(logits, target)
    aux_loss, aux_items = aux_criterion(aux, target)
    total_loss = main_loss + aux_loss
```

更干净的做法：  
新配置下不再实例化 `AuxiliaryLoss`，所有 aux 都由 `CompositeForensicLoss` 统一管理。

---

## 11. __init__.py 修改

文件：

```text
src/losses/__init__.py
```

新增导出：

```python
from .structured_forensic_loss import CompositeForensicLoss
```

---

## 12. trainer.py 修改

### 12.1 build loss

新增 helper：

```python
def build_loss(cfg):
    loss_cfg = cfg.get("loss", {}) or {}
    if str(loss_cfg.get("type", "")).lower() == "composite_forensic":
        from src.losses import CompositeForensicLoss
        return CompositeForensicLoss(loss_cfg), None
    return SegmentationLoss(loss_cfg), AuxiliaryLoss(cfg.get("aux_loss", {}))
```

在 `run_train()` 中替换：

```python
criterion = SegmentationLoss(cfg.get("loss", {})).to(device)
aux_criterion = AuxiliaryLoss(cfg.get("aux_loss", {})).to(device)
```

为：

```python
criterion, aux_criterion = build_loss(cfg)
criterion = criterion.to(device)
if aux_criterion is not None:
    aux_criterion = aux_criterion.to(device)
```

### 12.2 训练 step

替换：

```python
main_loss, main_items = criterion(logits, target)
aux_loss, aux_items = aux_criterion(aux, target)
total_loss = main_loss + aux_loss
```

为：

```python
if aux_criterion is None:
    total_loss, loss_items = criterion(logits, target, aux=aux, epoch=epoch)
    main_loss = total_loss
    aux_loss = total_loss * 0.0
    main_items = loss_items
    aux_items = {}
else:
    main_loss, main_items = criterion(logits, target)
    aux_loss, aux_items = aux_criterion(aux, target)
    total_loss = main_loss + aux_loss
```

### 12.3 evaluate

evaluate 同样改：

```python
if aux_criterion is None:
    total_loss, loss_items = criterion(logits, target, aux=aux, epoch=None)
else:
    loss, _ = criterion(logits, target)
    aux_loss, _ = aux_criterion(aux, target)
    total_loss = loss + aux_loss
```

---

## 13. optimizer.py 修改

新增 temporal encoder 参数组。

如果当前 optimizer 已按 keyword 区分：

```text
lora
ccm
fgm
decoder
```

请新增：

```text
temporal_fusion
temporal_encoder
```

推荐配置：

```yaml
optimizer:
  learning_rate: 1.0e-4
  lr_lora: 1.0e-5
  lr_temporal_encoder: 5.0e-5
  lr_ccm: 1.0e-4
  lr_fgm: 0.0
  lr_decoder: 1.0e-4
  weight_decay: 1.0e-4
```

参数匹配建议：

```python
if "temporal_fusion" in name or "temporal_encoder" in name or "ttf" in name:
    lr = lr_temporal_encoder
```

注意：  
TTF 是新模块，学习率可以高于 LoRA，但不要高于 decoder 太多。推荐 `5e-5` 起步。

---

## 14. 新配置文件

创建：

```text
configs/b23_ttf_ccm_structloss_lora32.yml
```

内容参考：

```yaml
type: train
seed: 666666

input_size: 512
batch_size: 1
num_clips: 4
num_frames: 4
clip_stride: 1
encoder_chunk: 0
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
appearance_augment_prob: 0.5

temporal_augment:
  frame_swap:
    enabled: true
    prob: 0.10
    max_swaps: 1
    local_radius: 2
  frame_drop:
    enabled: true
    prob: 0.10
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

temporal_encoder:
  enabled: true
  type: "patch_token_fusion"
  insert_position: "after_patch_embed_before_blocks"
  token_fusion:
    enabled: true
    mode: "same_patch_temporal_attention"
    dim: 1024
    bottleneck_dim: 128
    heads: 4
    dropout: 0.05
    frame_mask: "bidirectional"
    alpha_init: 0.0
    proj_zero_init: true
    return_residual_energy: true

lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.1
  layers: "all"
  targets: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"

tfcu:
  version: "ttf_ccm_lite"
  ccm:
    enabled: true
    dim: 128
    heads: 4
    q_resolution: 32
    kv_resolution: 16
    frame_mask: "lower_triangular"
    random_mask: {enabled: true, keep_prob: 0.7}
    fusion: "residual_concat"
    alpha_init: 0.002
    fuse_zero_init: true
    aux_head: true

  fgm:
    enabled: false
    cue_dim: 64
    cue_resolution: 16
    bank_len: 0
    detach_bank: true
    propagation: {enabled: false}
    aggregation: {enabled: false}
    prompt: {enabled: false}
    aux_head: false

fusion:
  type: "concat_fuse"
  static_channels: 128
  ccm_channels: 64
  fgm_channels: 64
  out_channels: 128
  use_depthwise_refine: true

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
  boundary_head: {enabled: true, resolution: 128}

fgm_bank:
  train_full_video_windows: false
  stateful_train: false
  stateful_eval: false
  reset_on_new_video: false
  detach_cross_window: true

val_full_video: false
test_full_video: false
test_max_clips: 4

loss:
  type: "composite_forensic"

  region:
    enabled: true
    weight: 1.0
    focal_bce:
      weight: 0.6
      gamma: 2.0
      alpha: 0.75
      adaptive_pos_weight: true
      pos_weight_clip: [1.0, 20.0]
    focal_tversky:
      weight: 0.4
      alpha: 0.3
      beta: 0.7
      gamma: 0.75
      smooth: 1.0e-6

  boundary:
    enabled: true
    weight: 0.20
    kernel_size: 5
    band_width: 5
    bce_weight: 0.7
    dice_weight: 0.3

  pairwise_temporal:
    enabled: true
    weight: 0.08
    warmup_epochs: 20
    max_pairs: 24
    downsample: 128
    positive_delta_weight: 1.0
    negative_delta_weight: 0.35
    loss_type: "charbonnier"

  deep_supervision:
    enabled: true
    weight: 0.25
    normalize_by_active_heads: true
    heads:
      mask128: {enabled: true, weight: 0.50}
      mask256: {enabled: true, weight: 0.25}
      ccm_mask32: {enabled: true, weight: 0.25}
      fgm_mask32: {enabled: false, weight: 0.0}
      boundary128: {enabled: true, weight: 0.25}

  token_separation:
    enabled: false
    weight: 0.03
    warmup_epochs: 50
    min_pixels_per_class: 8
    max_tokens_per_frame: 256
    margin: 0.20

  ttf_regularization:
    enabled: false
    weight: 0.005
    warmup_epochs: 20
    nochange_weight: 1.0

optimizer:
  learning_rate: 1.0e-4
  lr_lora: 1.0e-5
  lr_temporal_encoder: 5.0e-5
  lr_ccm: 1.0e-4
  lr_fgm: 0.0
  lr_decoder: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  type: "cosine"
  warmup_epochs: 10
  min_lr: 1.0e-6

train:
  n_epochs: 1000
  save_dir: "runs/b23_ttf_ccm_structloss_lora32"
  val_interval: 10
  max_grad_norm: 1.0
  skip_nonfinite: true

ddp:
  auto_torchrun: true
  cuda_visible_devices: "4,5"
  nproc_per_node: 2
  dist_backend: "nccl"
  find_unused_parameters: false
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  torchrun_log_dir: "runs/b23_ttf_ccm_structloss_lora32/torchrun_logs"
  torchrun_tee: ""

debug:
  log_shapes: true
```

---

## 15. 原始分辨率切片 / RelayFormer-style 扩展

本轮实现建议先完成 512 TTF。  
但请预留接口支持后续 original-resolution local units。

### 15.1 为什么不直接原图进 DINOv3

DINOv3 ViT-L/16 在 512 输入下 patch 数是：

```text
32 * 32 = 1024
```

如果输入 1080p，patch 数接近：

```text
68 * 120 = 8160
```

self-attention 复杂度近似按 `N^2` 增长，显存和速度会大幅不可控。

### 15.2 Relay-style 正确接法

不是整图输入，而是：

```text
原始视频帧 [B,T,3,H0,W0]
        |
        | 切 512x512 overlapping local units
        v
units [B,U,T,3,512,512]
        |
        | 每个 unit 内跑 TTF-DINOv3
        v
unit logits [B,U,T,1,512,512]
        |
        | overlap weighted stitching
        v
full logits [B,T,1,H0,W0]
```

后续可加 relay tokens：

```text
local patch tokens <-> relay tokens <-> global unit/time exchange <-> local patch tokens
```

### 15.3 本轮预留配置，不启用

```yaml
relay_units:
  enabled: false
  unit_size: 512
  stride: 384
  overlap_weight: "hann"
  relay_tokens: 4
  relay_layers: [5, 11, 17, 23]
  relay_alpha_init: 0.0
```

本轮不要同时实现复杂 relay tokens，否则很难判断 TTF 和 loss 的贡献。

---

## 16. 必须完成的测试

### 16.1 shape test

新增脚本：

```text
scripts/debug_ttf_forward.py
```

测试：

```python
video = torch.randn(1, 4, 4, 3, 512, 512).cuda()
model = B23TFCUCCMFGMLiteModel(cfg).cuda()
out = model(video)
assert out["logits"].shape == (1, 4, 4, 1, 512, 512)
```

### 16.2 temporal disabled 等价测试

当：

```yaml
temporal_encoder.enabled: false
```

必须走旧 encoder。旧配置不受影响。

当：

```yaml
temporal_encoder.enabled: true
token_fusion.alpha_init: 0.0
token_fusion.proj_zero_init: true
```

理论上 TTF residual 初始为 0。  
但因为手写 DINOv3 forward 可能和 `get_intermediate_layers` 在细节上有差异，需要做数值检查：

```text
same frames
old encoder output vs new encoder output with TTF zero-init
mean abs diff 应尽量接近 0
如果不接近，优先检查：
1. cls/storage token 切分
2. rope_sincos 调用
3. norm 时机
4. output_block index 是否一致
```

### 16.3 顺序无关测试

新模型没有 FGM bank，因此对 batch/window 顺序不敏感。

测试：

```text
取同一批 windows
order A 跑一遍
order B 打乱跑一遍
按 sample id 对齐 logits
结果应一致，允许浮点误差
```

### 16.4 loss 数值稳定测试

测试：

```python
logits = torch.randn(1,4,4,1,512,512).cuda()
target = torch.randint(0,2,(1,4,4,1,512,512)).float().cuda()
loss, items = criterion(logits, target, aux={}, epoch=0)
assert torch.isfinite(loss)
```

还要测试空 mask：

```python
target.zero_()
loss must be finite
```

以及全前景 mask：

```python
target.fill_(1)
loss must be finite
```

---

## 17. 消融实验计划

必须至少跑以下实验：

### 17.1 baseline

```text
configs/b23_ccm_lite_lora32.yml
```

### 17.2 只加 TTF，旧 loss

```text
b23_ttf_ccm_lite_lora32_legacy_loss.yml
```

目的：单独验证 TTF 是否有效。

### 17.3 不加 TTF，只用新 loss

```text
b23_ccm_structloss_lora32.yml
```

目的：验证 loss 重构本身是否有效。

### 17.4 TTF + 新 loss

```text
b23_ttf_ccm_structloss_lora32.yml
```

主实验。

### 17.5 TTF attention mask 消融

```text
frame_mask: "bidirectional"
frame_mask: "lower_triangular"
```

如果 bidirectional 跨源更好，说明不用模拟在线 causality；  
如果 lower_triangular 更好，说明模型受益于更保守的时序先验。

### 17.6 pairwise temporal loss 权重消融

```text
weight: 0.00
weight: 0.04
weight: 0.08
weight: 0.12
```

重点看 OPN，不要只看 DVI/CPNET。

---

## 18. 训练命令

```bash
bash scripts/train_ddp.sh configs/b23_ttf_ccm_structloss_lora32.yml
```

快速 debug：

```bash
python scripts/debug_ttf_forward.py --config configs/b23_ttf_ccm_structloss_lora32.yml
```

测试：

```bash
python test.py --config configs/b23_ttf_ccm_structloss_lora32.yml \
  --checkpoint runs/b23_ttf_ccm_structloss_lora32/best_iou.pt
```

---

## 19. 实现完成标准

agent 完成后必须满足：

1. `b23_ccm_lite_lora32.yml` 原配置可以继续训练，不报错。
2. 新配置 `b23_ttf_ccm_structloss_lora32.yml` 可以 forward。
3. 新模型输出：
   - `logits: [B,M,K,1,512,512]`
   - `aux.mask128`
   - `aux.mask256`
   - `aux.ccm_mask32`
   - `aux.debug.temporal_encoder`
4. 新 loss 对正常 mask、空 mask、全前景 mask 都是 finite。
5. 新数据流不要求 stateful FGM bank。
6. DDP 下不需要 `VideoWindowSampler` 来保持视频窗口顺序。
7. 训练日志能显示：
   - `region_loss`
   - `boundary_loss`
   - `pairwise_temporal_loss`
   - `deep_supervision_loss`
   - `ttf_alpha`
8. 至少保存：
   - `runs/b23_ttf_ccm_structloss_lora32/config.resolved.json`
   - `runs/b23_ttf_ccm_structloss_lora32/log.csv`
   - `runs/b23_ttf_ccm_structloss_lora32/best_iou.pt`

---

## 20. 关键风险与规避

### 风险 1：手写 DINOv3 forward 和原始接口不一致

规避：

- 先做 old/new encoder zero-init 等价测试。
- 检查 cls/storage token 数量。
- 检查 rope 调用。
- 检查 block index。
- 检查 norm 时机。

### 风险 2：TTF 过拟合同源时序 cue

规避：

- `alpha_init=0.0`
- `proj_zero_init=true`
- `lr_temporal_encoder=5e-5`
- pairwise temporal loss 权重从 0.04 或 0.08 起步
- 不启用 FGM bank
- checkpoint 选择不要只看 DVI/CPNET，必须看 OPN 或泛化分数

### 风险 3：新 loss 过强，导致 mask 过平滑

规避：

- boundary weight 不超过 0.20
- pairwise temporal weight 不超过 0.12
- temporal loss 使用 warmup
- negative_delta_weight 小于 positive_delta_weight

### 风险 4：空 mask 下 focal/tversky 不稳定

规避：

- 所有分母加 smooth/eps
- adaptive pos_weight clamp
- 空前景时 region loss 仍然主要由 BCE 负责
- token_separation 对没有前景或背景 token 的 batch 直接返回 0

---

## 21. 推荐第一版落地范围

第一版必须实现：

```text
DINOv3B23TemporalEncoder
TemporalPatchTokenFusion
CompositeForensicLoss:
    region
    boundary
    pairwise_temporal
    deep_supervision
new config
trainer compatibility
optimizer param group
debug forward script
```

第一版暂不启用：

```text
token_separation
ttf_regularization
original-resolution relay units
FGM
```

第一版目标不是一次把所有指标推满，而是验证：

```text
TTF 是否能在不依赖 full-video stateful bank 的情况下，
让 DINOv3 的 B23 特征提前感知局部跨帧 inpainting 异常，
并且在 OPN 这类跨源测试上不下降。
```

---

## 22. 最终期望结构

```text
B23-TTF-CCM-Lite-LoRA32

Input [B,M,K,3,512,512]
        |
        v
DINOv3 patch embedding
        |
        v
Temporal Patch Token Fusion
        |
        v
DINOv3 blocks 0..23
        |
        v
B23 feature [B,M,K,1024,32,32]
        |
        v
CCM-Lite
        |
        v
Low-res fusion
        |
        v
LiteBoundaryDecoder
        |
        v
Mask logits [B,M,K,1,512,512]
```

对应训练目标：

```text
CompositeForensicLoss =
    region segmentation
  + boundary localization
  + pairwise order-invariant temporal consistency
  + normalized deep supervision
  + optional token anomaly separation
```

这条路线比继续强化 FGM 更适合跨源泛化，因为它不把历史 window cue 存成 memory，而是在每个输入 window 内用轻量、局部、可并行的 token 融合把跨帧异常提示注入 DINOv3。
