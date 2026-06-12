# 当前四个实验版本对比

本文档对应 `runs/` 下当前已经跑过的四个实验：

```text
runs/b23_ccm_lite_lora32
runs/b23_ccm_fgm_lite_lora32
runs/b23_ccm_fgm_lite_lora32_more
runs/b23_ccm_fgm_forensic_gated_lora32
```

所有结果均来自各自 `log.txt` 和 `config.resolved.json`，门控参数来自对应 `best_iou.pt`。

## 共同点

四个版本都基于同一个轻量框架：

```text
DINOv3 ViT-L/16 B23 feature
+ LoRA rank 32
+ Low-res temporal/fusion module
+ LiteBoundaryDecoder
```

共同训练/评估设置大体一致：

```text
input_size: 512
batch_size: 1
num_frames: 4
backbone: DINOv3 ViT-L/16 B23, 32x32 feature
train_samples: DAVIS-VI_tra_DVI_30 + DAVIS-VI_tra_CPNET_30
val_samples: DAVIS-VI_val_DVI_20 + DAVIS-VI_val_CPNET_20
main loss: dice + bce + tversky + boundary
optimizer: AdamW, base lr 1e-4
scheduler: cosine
```

差异主要来自是否启用 FGM、FGM bank 长度、CCM 初始强度、是否加入 forensic residual branch、是否使用 quality-gated bank。

## 总览表

| Run | 结构 | Best Epoch | Best Val IoU | Best Val F1 | Last Val IoU | 结论 |
|---|---|---:|---:|---:|---:|---|
| `b23_ccm_lite_lora32` | CCM-only，无 FGM | 599 | 0.8218 | 0.8974 | 0.8127 | 稳定，F1 最高，训练很长 |
| `b23_ccm_fgm_lite_lora32` | CCM + FGM，bank=3 | 229 | 0.8234 | 0.8925 | 0.8196 | 稳定，IoU 略高于 CCM-only |
| `b23_ccm_fgm_lite_lora32_more` | 更强 CCM + FGM，bank=5 | 219 | 0.8253 | 0.8934 | 0.8119 | 最佳 IoU，但后续出现 NaN 风险 |
| `b23_ccm_fgm_forensic_gated_lora32` | CCM + FGM + forensic branch + quality gate | 129 | 0.7719 | 0.8521 | 0.7719 | 当前版本明显退化，不建议作为主结果 |

注意：`b23_ccm_lite_lora32` 的 `config.resolved.json` 中 `num_clips` 与 `log.txt` 头部记录不完全一致，可能经历过继续训练或配置覆盖；本表以 `log.txt` 的训练结果为准。

## 版本 1：`b23_ccm_lite_lora32`

### 结构

这是 CCM-only 消融版本：

```text
B23 feature
  -> CCM-Lite masked cross-attention
  -> Low-res fusion
  -> LiteBoundaryDecoder
```

FGM 关闭：

```yaml
tfcu:
  ccm:
    enabled: true
  fgm:
    enabled: false
```

无历史 bank、无跨 window 记忆、无 forensic branch。

### 训练结果

日志共记录到 epoch 999，验证次数 100。最佳点：

```text
best epoch: 599
val_loss: 0.2393
val_iou: 0.8218
val_f1: 0.8974
val_precision: 0.8748
val_recall: 0.9315
val_accuracy: 0.9873
```

最后一次验证：

```text
epoch: 999
val_loss: 0.2355
val_iou: 0.8127
val_f1: 0.8922
```

best checkpoint 中：

```text
ccm.alpha_cc = 0.1035
```

### 评价

这个版本非常稳定，没有 NaN，F1 最高。缺点是训练周期很长，且只依赖 clip 内 CCM，没有显式历史记忆。作为 baseline 很可靠，也适合做“无 FGM 时序记忆”的消融参考。

## 版本 2：`b23_ccm_fgm_lite_lora32`

### 结构

这是标准 CCM + FGM 版本：

```text
B23 feature
  -> CCM-Lite
  -> FGM-Lite historical cue propagation
  -> Low-res concat fusion
  -> LiteBoundaryDecoder
```

关键配置：

```yaml
num_clips: 2
num_frames: 4

fgm_bank:
  train_full_video_windows: true
  stateful_train: true
  stateful_eval: true
  reset_on_new_video: true
  detach_cross_window: true

tfcu:
  ccm:
    alpha_init: 0.002
  fgm:
    bank_len: 3
    propagation:
      hist_len: 2
      topk: 128
```

FGM bank 存储 16x16 的 historical forgery cue，跨同一视频的 window 延续，跨视频 reset。

### 训练结果

日志记录到 epoch 251，验证次数 42。最佳点：

```text
best epoch: 229
val_loss: 0.2384
val_iou: 0.8234
val_f1: 0.8925
val_precision: 0.8750
val_recall: 0.9191
val_accuracy: 0.9863
```

最后一次验证：

```text
epoch: 249
val_loss: 0.2439
val_iou: 0.8196
val_f1: 0.8906
```

best checkpoint 中：

```text
ccm.alpha_cc = 0.0307
fgm.cue_feedback_scale = 0.0624
```

### 评价

标准 FGM 版本比 CCM-only 的最佳 IoU 略高：

```text
0.8234 vs 0.8218
```

但 F1 略低：

```text
0.8925 vs 0.8974
```

整体稳定，没有 NaN。它说明 FGM 的历史 cue 对 IoU 有帮助，但收益不大。这个版本适合做主实验的稳定候选。

## 版本 3：`b23_ccm_fgm_lite_lora32_more`

### 结构

这是更强的 CCM + FGM 版本。相对标准 FGM，主要增强如下：

```yaml
tfcu:
  ccm:
    alpha_init: 0.01
  fgm:
    bank_len: 5
```

也就是：

```text
更大的 CCM 初始残差权重
更长的历史 cue bank
```

FGM propagation 仍使用 `hist_len=2`，但 bank 能保存更长视频上下文。

### 训练结果

日志记录到 epoch 231，验证次数 26。最佳点：

```text
best epoch: 219
val_loss: 0.2352
val_iou: 0.8253
val_f1: 0.8934
val_precision: 0.8924
val_recall: 0.9017
val_accuracy: 0.9869
```

后续记录中出现过 NaN：

```text
epoch 239: val_loss nan, val_iou 0.0000
epoch 249: val_loss nan, val_iou 0.0000
```

但当前 `best_iou.pt` 是干净的。best checkpoint 中：

```text
ccm.alpha_cc = 0.0345
fgm.cue_feedback_scale = 0.0681
```

### 评价

这是四个实验里验证 IoU 最好的版本：

```text
val_iou = 0.8253
```

它比标准 FGM 多约 0.19 个点 IoU，比 CCM-only 多约 0.36 个点 IoU。优势不大，但确实是当前最高。问题是后续训练出现 NaN，说明更强的 bank/CCM 设置带来了稳定性风险。

建议：

```text
同源 DVI/CPNET 结果优先使用 best_iou.pt
不要使用 NaN 污染后的 latest.pt
不建议无控制地继续训练
```

## 版本 4：`b23_ccm_fgm_forensic_gated_lora32`

### 结构

这是泛化增强尝试版本：

```text
B23 high-level feature
  -> CCM-Lite
  -> FGM-Lite
  -> quality-gated FGM bank
  -> residual/noise forensic branch
  -> LiteBoundaryDecoder
```

新增模块：

```yaml
forensic_branch:
  enabled: true
  alpha_init: 0.01
  out_channels: 32

fgm_bank:
  quality_gate:
    enabled: true
    keep_threshold: 0.20
```

同时相对标准 FGM 做了保守化：

```yaml
tfcu:
  fgm:
    bank_len: 3
    aggregation:
      diff_scale: 0.75
      feedback_init: 0.005
    propagation:
      topk: 96

aux_loss:
  fgm_mask32:
    weight: 0.03
```

### 训练结果

日志当前记录到 epoch 136，验证次数 13。最佳点：

```text
best epoch: 129
val_loss: 0.2843
val_iou: 0.7719
val_f1: 0.8521
val_precision: 0.8202
val_recall: 0.9282
val_accuracy: 0.9781
```

top 验证点：

```text
epoch 129: val_iou 0.7719
epoch 69 : val_iou 0.7708
epoch 29 : val_iou 0.7462
```

best checkpoint 中：

```text
ccm.alpha_cc = 0.0656
fgm.cue_feedback_scale = 0.0479
forensic_branch.alpha = 0.0933
```

### 评价

这个版本当前明显退化：

```text
best val_iou 0.7719
比标准 FGM 低约 5.15 个点
比 more FGM 低约 5.34 个点
```

主要原因很可能是：

```text
forensic_branch.alpha 从 0.01 增长到 0.0933，低层分支干预过强
ccm.alpha_cc 增长到 0.0656，高于标准 FGM/more 的 0.03 左右
FGM 本身被削弱：topk=96、diff_scale=0.75、fgm aux weight=0.03
quality_gate 阈值 0.20 太低，未能有效筛掉低质量 cue
```

因此，该版本只能作为失败消融或后续改进基础，不建议作为主结果。

## 横向结论

### 当前最佳主结果

如果只看 DVI/CPNET 验证 IoU：

```text
b23_ccm_fgm_lite_lora32_more/best_iou.pt
```

最佳指标：

```text
val_iou: 0.8253
val_f1 : 0.8934
```

但它后续出现过 NaN，因此必须只使用 `best_iou.pt`，不要使用被 NaN 污染的后续 checkpoint。

### 当前最稳 baseline

```text
b23_ccm_lite_lora32/best_iou.pt
```

它没有 FGM，训练稳定，F1 最高：

```text
val_iou: 0.8218
val_f1 : 0.8974
```

适合作为强 baseline 和消融对照。

### FGM 是否有效

FGM 有效，但收益较小：

```text
CCM-only best IoU: 0.8218
Standard FGM best IoU: 0.8234
More FGM best IoU: 0.8253
```

FGM 提升 IoU，但 F1 不一定超过 CCM-only。说明历史 cue 对 mask 区域覆盖有帮助，但可能带来一些 precision/recall 平衡变化。

### Forensic-gated 版本是否有效

当前无效。它的设计目标是提高泛化，但在 DVI/CPNET 验证集上已经明显落后。主要问题不是训练不足，而是 forensic branch 和 CCM 干预过强，同时 FGM 被削弱。

## 后续建议

1. 主线结果使用 `b23_ccm_fgm_lite_lora32_more/best_iou.pt` 或 `b23_ccm_fgm_lite_lora32/best_iou.pt`。
2. 如果追求稳定和 F1，保留 `b23_ccm_lite_lora32/best_iou.pt` 作为强 baseline。
3. 不继续使用当前 `b23_ccm_fgm_forensic_gated_lora32` 作为主实验。
4. 如果要重做 forensic-gated v2，建议：

```yaml
forensic_branch:
  alpha_init: 0.002

optimizer:
  lr_forensic: 1.0e-5
  lr_ccm: 2.0e-5
  lr_fgm: 5.0e-5

tfcu:
  fgm:
    bank_len: 5
    aggregation:
      diff_scale: 1.0
    propagation:
      topk: 128

fgm_bank:
  quality_gate:
    keep_threshold: 0.45
```

5. 后续日志建议额外记录：

```text
ccm.alpha_cc
fgm.cue_feedback_scale
forensic_branch.alpha
fgm_bank_quality_keep
fgm_bank_quality
```

这样能更早发现某个门控过强或 bank 被污染。
