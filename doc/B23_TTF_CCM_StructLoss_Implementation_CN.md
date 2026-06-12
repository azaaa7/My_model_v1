# B23-TTF-CCM-StructLoss 实现说明

本文档对应新配置：

```bash
configs/b23_ttf_ccm_structloss_lora32.yml
```

该版本是在 `b23_ccm_lite_lora32.yml` 的强泛化 baseline 上新增的独立实验版本，不覆盖、不删除已有配置。

## 1. 模型结构

整体结构：

```text
video [B,M,K,3,512,512]
  -> DINOv3 patch embedding
  -> TemporalPatchTokenFusion
  -> DINOv3 blocks 0..23
  -> B23 feature [B*M*K,1024,32,32]
  -> CCM-Lite
  -> low-res fusion
  -> LiteBoundaryDecoder
  -> logits [B,M,K,1,512,512]
```

### 1.1 TemporalPatchTokenFusion

新增文件：

```text
src/models/tfcu/temporal_token_fusion.py
```

输入输出：

```python
x:     [B, M, K, N, C]
x_out: [B, M, K, N, C]
```

其中 `N=32*32`，`C=1024`。

它只在同一个空间 patch 位置上做跨帧 attention：

```text
x[b,m,:,n,:] -> temporal attention over K
```

不做全空间 attention，也不使用 full-video bank，因此不会引入 FGM 那种跨窗口状态。

初始化策略：

- `alpha_init: 0.0`
- `proj_zero_init: true`

所以刚开始 `x_out == x`，不会突然破坏原 DINOv3 token。

### 1.2 DINOv3B23TemporalEncoder

新增文件：

```text
src/models/dinov3_b23_temporal_encoder.py
```

它不再调用黑盒：

```python
backbone.get_intermediate_layers(...)
```

而是手动展开：

```text
normalize
prepare_tokens_with_masks
split cls/storage tokens and patch tokens
TTF on patch tokens
concat tokens
DINOv3 blocks
norm
reshape patch tokens
```

输出严格为：

```python
[B*M*K, 1024, 32, 32]
```

`use_activation_checkpoint: true` 会传入 temporal encoder，并在 DINOv3 blocks 内逐层 checkpoint，以降低 4 clips x 4 frames 的显存压力。

## 2. 和旧版本的隔离

主模型只在配置中启用：

```yaml
temporal_encoder:
  enabled: true
```

时才使用 `DINOv3B23TemporalEncoder`。

旧配置没有该字段或为 false 时，仍然走原来的：

```python
DINOv3B23Encoder
```

因此旧实验、旧权重、旧训练命令不受影响。

## 3. 新配置关键项

新配置路径：

```text
configs/b23_ttf_ccm_structloss_lora32.yml
```

核心开关：

```yaml
num_clips: 4
num_frames: 4
encoder_chunk: 0

temporal_encoder:
  enabled: true

tfcu:
  version: "ttf_ccm_lite"
  ccm:
    enabled: true
  fgm:
    enabled: false

fgm_bank:
  stateful_train: false
  stateful_eval: false

val_full_video: false
test_full_video: false
```

含义：

- 使用 4 个 clip、每个 clip 4 帧，让 TTF 在当前 window 内学习跨帧 token 关系。
- 关闭 FGM 和 stateful bank，避免当前版本同时引入历史记忆变量。
- 验证和测试使用轻量 window 流程，不跑 full-video bank。

## 4. 结构化 Loss

当前配置已进一步简化为论文式最小监督目标：

```text
L_total = L_seg + lambda * ramp(epoch) * L_tda
L_seg = 0.5 * L_focal_bce + 0.5 * L_dice
```

启用的新 loss 为：

```text
src/losses/ttf_minimal_loss.py
TTFMinimalLoss
```

`L_seg` 只监督最终 decoder 输出的 mask logits，不使用 aux head。`L_tda` 也只使用最终 mask prediction，在当前输入 window 内随机采样帧对做 Temporal Difference Alignment，不使用 token、不使用 FGM bank、不跨 window 存历史状态。

旧的 `CompositeForensicLoss` 代码仍保留，供之前的结构化 loss 消融继续使用；但 `configs/b23_ttf_ccm_structloss_lora32.yml` 当前不再启用它。

<!-- Legacy notes below describe the previous structured-loss implementation kept in code. -->

新增文件：

```text
src/losses/structured_forensic_loss.py
```

新 loss：

```python
CompositeForensicLoss
```

配置入口：

```yaml
loss:
  type: "composite_forensic"
```

包含四类主要监督：

```text
region_loss
boundary_loss
pairwise_temporal_loss
deep_supervision_loss
```

### 4.1 region_loss

由 Balanced Focal BCE 和 Focal Tversky 组成：

```text
region = 0.6 * focal_bce + 0.4 * focal_tversky
```

用于解决 inpainting mask 前景稀疏、漏检代价高的问题。

### 4.2 boundary_loss

从 GT 生成 boundary band：

```text
edge = dilate(mask) - erode(mask)
band = dilate(edge)
```

只在边界 band 内约束 BCE/Dice，避免边界 loss 在大面积背景上过强。

空 mask 或全前景 mask 没有有效边界时，该项返回 0，避免 NaN。

### 4.3 pairwise_temporal_loss

不依赖帧顺序，只在当前 `[M,K]` window 内取帧对：

```text
|pred_i - pred_j| vs |gt_i - gt_j|
```

默认最多采样 24 对，downsample 到 128 只用于训练/验证 loss，正式测试指标不使用该项。

### 4.4 deep_supervision_loss

统一管理 aux head：

```yaml
mask128
mask256
ccm_mask32
boundary128
```

并按 active head 权重归一化，避免打开更多 aux head 后总 loss 被动变大。

## 5. 测试阶段 GT 不下采样

`src/eval/tester.py` 调用：

```python
evaluate(..., include_aux_losses=False)
```

对于 `CompositeForensicLoss`，这意味着测试只计算主输出的 region/boundary loss：

- logits 如有需要会上采样到 GT 尺寸；
- GT 不会为了测试指标被下采样；
- pairwise temporal / deep supervision / token loss / TTF regularization 都不会在测试总结中启用。

因此测试指标仍然在原 GT 尺寸上计算。

## 6. 训练、调试、测试命令

训练：

```bash
bash scripts/train_ddp.sh configs/b23_ttf_ccm_structloss_lora32.yml
```

快速检查 loss 数值稳定性：

```bash
python scripts/debug_ttf_forward.py \
  --config configs/b23_ttf_ccm_structloss_lora32.yml \
  --skip-model \
  --device cpu \
  --loss-size 64
```

完整 forward shape 检查：

```bash
python scripts/debug_ttf_forward.py \
  --config configs/b23_ttf_ccm_structloss_lora32.yml
```

测试：

```bash
python test.py \
  --config configs/b23_ttf_ccm_structloss_lora32.yml \
  --checkpoint runs/b23_ttf_ccm_structloss_lora32/best_iou.pt
```

## 7. 当前验证结果

已完成轻量检查：

- `py_compile` 通过。
- `CompositeForensicLoss` 对随机 mask、空 mask、全前景 mask 均为 finite。
- `TemporalPatchTokenFusion` 输出 shape 正确。
- `alpha=0` 且 `proj_zero_init=true` 时，TTF 输出与输入最大差为 `0.0`。

尚未在本次实现中跑完整 DINOv3 forward，因为该检查会加载 ViT-L/16 权重并占用较多显存。需要时可用上面的完整 debug 命令单独运行。
