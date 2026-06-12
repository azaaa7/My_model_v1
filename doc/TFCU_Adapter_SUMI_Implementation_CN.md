# TFCU Adapter SUMI 版本说明

本文档对应 `doc/AGENT_IMPLEMENTATION_GUIDE_TFCU_ADAPTER_SUMI.md` 的落地实现。所有新功能默认关闭，旧的 CCM/FGM/FGM bank 实验配置不需要修改即可继续使用。

## 1. 新增结构

### 1.1 TaskSpecificForensicsAdapter

文件：`src/models/modules/task_forensics_adapter.py`

位置：DINOv3 B23 backbone 输出之后，进入 CCM/TCU/decoder 之前。

作用：

- 从 RGB 帧提取固定低层残差信号：灰度、Laplacian、Sobel、局部残差、梯度强度、帧间残差。
- 将低层 residual token 与 DINO 高层 feature 做 cross-attention。
- 使用低 alpha 和 gate 做保守残差注入：

```text
F_adapted = F + alpha * gate * adapter_delta
```

主要配置：

```yaml
task_forensics_adapter:
  enabled: true
  insertion_mode: "post_backbone"
  adapter_dim: 64
  alpha_init: 0.001
  alpha_max: 0.035
  drop_path: 0.20
  gate_bias_init: -2.3
  warmup_epochs: 10
```

输出 aux：

- `adapter_mask32`
- `adapter_boundary32`
- `adapter_gate`

日志会记录：

- `adapter_alpha`
- `adapter_gate_mean`
- `adapter_delta_norm`
- `adapter_mask32_loss`
- `adapter_boundary32_loss`

### 1.2 TemporalCueUnraveling

文件：`src/models/modules/temporal_cue_unraveling.py`

位置：CCM 之后，FGM 之前；新实验中 FGM 默认关闭，因此它相当于替代 raw FGM bank 的时序模块。

输入：

```text
features: [B, T, C, 32, 32]
lowres_logits: [B, T, 1, 32, 32]，可选，来自 CCM aux
```

三条分支：

- Momentary：相邻帧 feature diff + cosine change，捕捉突发篡改/不连续。
- Gradual：ConvGRU 累积短程变化，质量 gate 控制历史更新。
- Cumulative：前向/后向 EMA 聚合长程趋势。

输出：

- `feat_delta`
- `logit_delta32`
- `tcu_momentary_mask32`
- `tcu_gradual_mask32`
- `tcu_cumulative_mask32`
- `gate`
- `quality`

主要配置：

```yaml
temporal_cue_unraveling:
  enabled: true
  hidden_dim: 128
  alpha_init: 0.001
  alpha_max: 0.030
  branch_dropout: 0.10
  gradual:
    detach_history: true
    min_quality: 0.25
  cumulative:
    momentum: 0.90
```

日志会记录：

- `tcu_alpha`
- `tcu_gate_momentary_mean`
- `tcu_gate_gradual_mean`
- `tcu_gate_cumulative_mean`
- `tcu_quality_mean`
- `tcu_momentary_loss`
- `tcu_gradual_loss`
- `tcu_cumulative_loss`

### 1.3 SUMI-style losses

文件：`src/losses/sumi_localization_losses.py`

包括三类约束：

- Sufficiency：要求 adapter/TCU/CCM 等各视角单独也能定位 mask。
- Minimality：IB KL 约束，降低无关信息。
- Source adversarial：GRL 源域分类，让特征尽量不携带 DVI/CPNET/OPN 来源偏差。
- Background suppression：抑制背景区域激活。

主要配置：

```yaml
loss:
  sumi:
    enabled: true
    sufficiency:
      start_epoch: 5
      adapter_mask32: 0.020
      tcu_gradual_mask32: 0.025
    minimality:
      start_epoch: 20
      ib_kl_weight: 0.0002
    source_adversarial:
      enabled: true
      start_epoch: 30
      weight: 0.010
      grl_lambda: 0.05
```

日志会记录：

- `sumi_sufficiency_loss`
- `sumi_ib_kl`
- `sumi_source_adv_loss`
- `background_suppression_loss`
- `sumi_loss`

## 2. 新增实验配置

四个配置均位于 `configs/experiments/`。

### E1: Adapter baseline

```bash
bash scripts/train_ddp.sh configs/experiments/b23_task_adapter_baseline_lora32.yml
```

结构：

```text
DINOv3 B23 + LoRA -> TaskSpecificForensicsAdapter -> Static decoder path
```

关闭：

- CCM
- FGM
- TCU
- SUMI

目的：先验证任务专用低层 residual adapter 是否能提升无时序 baseline。

### E2: SUMI losses

```bash
bash scripts/train_ddp.sh configs/experiments/b23_sumi_losses_lora32.yml
```

结构：

```text
DINOv3 B23 + LoRA -> CCM -> decoder + SUMI losses
```

关闭：

- FGM bank
- Task adapter
- TCU

目的：单独验证 SUMI-style sufficiency/minimality 对泛化是否有帮助。

### E3: CCM + TCU unravel

```bash
bash scripts/train_ddp.sh configs/experiments/b23_ccm_tfcu_unravel_lora32.yml
```

结构：

```text
DINOv3 B23 + LoRA -> CCM -> TemporalCueUnraveling -> decoder
```

关闭：

- FGM raw cue bank
- Task adapter
- SUMI

目的：用 dense momentary/gradual/cumulative 时序分解替代 FGM 记忆库。

### E4: Final

```bash
bash scripts/train_ddp.sh configs/experiments/b23_tfcu_adapter_sumi_final_lora32.yml
```

结构：

```text
DINOv3 B23 + LoRA
  -> TaskSpecificForensicsAdapter
  -> CCM
  -> TemporalCueUnraveling
  -> LiteBoundaryDecoder
  -> base loss + aux loss + SUMI loss
```

关闭：

- FGM raw cue bank
- RGFGM/HP3D/FPM 旧记忆模块

目的：兼顾同源精度和跨源泛化。

## 3. 轻量验证流程

无记忆库版本新增了验证覆盖配置：

```yaml
fgm_bank:
  stateful_eval: false
val_full_video: false
val_num_clips: 1
val_test_max_clips: 1
val_num_workers: 2
```

含义：

- 验证阶段不预构建整视频 window。
- 每个验证样本只抽一个 clip。
- 每个 clip 仍然有 `num_frames=4` 帧。
- 不携带 FGM bank，所以验证速度明显快于 full-video stateful eval。

测试阶段仍保留：

```yaml
test_full_video: true
test_max_clips: 2
```

因此正式测试可以继续使用较完整的视频覆盖。

## 4. 观察指标

训练后重点看这些日志字段：

```text
adapter_alpha
adapter_gate_mean
tcu_alpha
tcu_gate_momentary_mean
tcu_gate_gradual_mean
tcu_gate_cumulative_mean
sumi_sufficiency_loss
sumi_ib_kl
sumi_source_adv_loss
background_suppression_loss
```

测试对比建议表：

| Config | DVI IoU | CPNET IoU | OPN IoU | same_source_avg | pareto_score |
|---|---:|---:|---:|---:|---:|
| task_adapter_baseline | | | | | |
| sumi_losses | | | | | |
| ccm_tfcu_unravel | | | | | |
| tfcu_adapter_sumi_final | | | | | |

其中：

```text
same_source_avg = (DVI IoU + CPNET IoU) / 2
pareto_score = same_source_avg + 0.7 * OPN IoU
```

## 5. 稳定性建议

如果 final 版本训练不稳定或 OPN 降低，按顺序尝试：

1. 关闭 `loss.sumi.source_adversarial.enabled`。
2. 将 `task_forensics_adapter.alpha_max` 从 `0.035` 降到 `0.020`。
3. 将 `temporal_cue_unraveling.alpha_max` 从 `0.030` 降到 `0.020`。
4. 将 `loss.sumi.minimality.ib_kl_weight` 从 `0.0002` 降到 `0.0001`。
5. 将 `aux_loss.tcu_branch_diversity.weight` 暂时设为 `0.0`。

## 6. Loss 消融注意事项

可以直接注释部分主 loss、`aux_loss` 或 `loss.sumi` 做消融。例如只保留：

```yaml
loss:
  bce: {weight: 0.9}
  boundary: {weight: 0.10}
```

这种情况下模型里的 CCM aux head、decoder 中间监督 head、Adapter/TCU aux head 可能仍然会 forward。代码已经加入零权重依赖保护，因此这些 head 不会改变 loss 数值，也不会在 DDP 下触发 unused parameter 报错。

不建议为了这个问题打开：

```yaml
ddp:
  find_unused_parameters: true
```

除非你临时加入了新的分支但还没有接入 loss 或零权重依赖。默认保持 `false` 更快。

## 7. 测试阶段 GT 分辨率约束

正式 `test.py` 路径不允许对 GT 做下采样。当前测试流程为：

```text
dataset(test): 只 resize 输入图像到 input_size，不 resize GT mask
align_logits_masks: 将模型 logits 上采样到 GT 原始尺寸
metrics: 在 GT 原始尺寸上计算 IoU/F1/Precision/Recall
```

注意：`aux_loss` 和 `loss.sumi` 内部会为了监督 32/128 辅助头而 resize GT。为了避免测试阶段出现任何 GT 下采样，`src/eval/tester.py` 已经在调用 `evaluate()` 时设置：

```python
include_aux_losses=False
```

因此正式测试只计算主输出 loss 和原尺寸指标，不计算 aux/SUMI 测试 loss。
