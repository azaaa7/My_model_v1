# RGFGM + HP3D + FPM 新版本实现说明

对应配置：

```bash
configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml
```

训练命令：

```bash
bash scripts/train_ddp.sh configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml
```

这个版本保留原来的强基线 `DINOv3 ViT-L/16 B23 + LoRA32 + CCM-Lite + FGM-Lite + LiteBoundaryDecoder`，并新增三类保守增强模块：Reliability-Gated FGM、HP3D/Noise Adapter、Prototype Memory。旧配置没有这些字段时仍按旧路径运行。

## 1. 模型结构

主路径如下：

```text
RGB video window [B,M,K,3,512,512]
  -> DINOv3 ViT-L/16 第 23 层 B23 feature [B*M*K,1024,32,32]
  -> CCM-Lite clip 内时序相关增强
  -> FGM-Lite 历史 cue bank 传播
  -> ReliabilityGate 对 FGM 输出 f_ip 做空间门控
  -> LowResTFCUFusion 得到 [B*K,128,32,32]
  -> HP3D/Noise Adapter 以 capped alpha 注入小幅噪声残差
  -> Prototype Memory 读取前景/背景原型并注入小幅语义记忆残差
  -> LiteBoundaryDecoder 输出 mask/logit/boundary
```

新增模块是可开关的：

```yaml
reliability_gate.enabled: true
forensic_adapter.enabled: true
prototype_memory.enabled: true
```

## 2. Reliability-Gated FGM

FGM 本身仍负责从历史 cue bank 中传播时序信息，但现在会先经过一个可靠性门控：

```text
f_ip = f_ip * g_fgm
```

门控输入包括：

- FGM 辅助 mask 的 entropy
- FGM 辅助 mask 的 confidence
- HP3D/noise adapter 输出的 reliability map
- 预留 motion residual 通道，当前为 0

默认初始化很保守：

```yaml
gate_bias_init: -2.0
gate_max: 0.70
detach_inputs: true
```

训练日志会记录：

```text
train_reliability_gate_mean
train_reliability_gate_min
train_reliability_gate_max
```

如果 OPN 泛化下降明显，优先尝试 `A10_gate_bias_minus3`；如果同源数据提升不足，可以尝试 `A9_gate_bias_minus1`。

## 3. HP3D/Noise Adapter

这个模块不是复用之前失败的 `forensic_branch`，而是一个更小、更保守的低层噪声证据分支：

- 固定高通/SRM-like 2D filter
- 帧间 temporal high-pass
- 小型 2D projection
- gated additive residual
- `alpha` 使用硬上限 `CappedAlpha`

默认配置：

```yaml
forensic_adapter:
  enabled: true
  alpha_init: 0.001
  alpha_max: 0.020
  detach_input: true
  drop_path: 0.20
```

输出会以很小残差加到 decoder 前的 32x32 fusion feature：

```text
F = F + alpha_noise * g_noise * delta_noise
```

日志会记录：

```text
train_forensic_adapter_alpha
train_forensic_adapter_gate_mean
train_forensic_adapter_reliability_mean
```

正常情况下 `forensic_adapter_alpha` 不应超过配置的 `alpha_max`。如果看到默认配置下 alpha 大于 0.02，说明实现或 checkpoint 不匹配。

## 4. Prototype Memory

Prototype Memory 用前景/背景原型替代“只依赖 raw dense cue bank”的记忆方式。它挂在现有 FGM bank 对象上，随视频窗口流转，并在新视频时随 bank reset。

每个视频状态中维护：

```text
fg_proto: [num_fg_proto, 128]
bg_proto: [num_bg_proto, 128]
fg_valid/bg_valid
```

写入条件很严格：

```yaml
write_confidence_min: 0.75
write_entropy_max: 0.35
write_area_min: 0.002
write_area_max: 0.60
write_gate_min: 0.15
write_warmup_epochs: 10
```

满足条件时，用当前低分辨率 feature 和预测 mask 概率计算前景/背景加权平均，再用 EMA 更新原型；写入不反传，避免跨窗口反传图爆显存。
训练前 10 个 epoch 默认不写入 prototype，避免 early prediction 噪声污染整视频记忆；验证/测试阶段不受这个 warmup 限制。

读取时用当前 32x32 feature 查询 fg/bg prototype，得到 `delta_proto` 后经保守 read gate 注入：

```yaml
read_gate_bias_init: -1.5
read_gate_max: 0.60
```

默认新配置使用：

```yaml
prototype_memory:
  enabled: true
  replace_raw_bank: true
```

这表示 Prototype Memory 是主要跨窗口记忆；旧 raw FGM cue bank 代码仍保留，旧配置不受影响。

日志会记录：

```text
train_prototype_memory_write_rate
train_prototype_memory_read_gate_mean
train_prototype_memory_valid
train_prototype_memory_fg_proto_norm
train_prototype_memory_bg_proto_norm
```

## 5. Loss 和优化器

新配置新增了低层噪声监督和门控正则：

```yaml
aux_loss:
  noise_mask32: {enabled: true, weight: 0.015}
  noise_boundary: {enabled: true, weight: 0.020}
  gate_regularization: {enabled: true, weight: 0.002, target: 0.0}
```

优化器新增独立学习率分组：

```yaml
lr_reliability_gate: 2.0e-5
lr_prototype_memory: 1.0e-5
lr_forensic_adapter: 1.0e-5
```

这样低层噪声和 memory readout 不会用 decoder 的大 lr 乱跑。

## 6. 稳定性保护

新配置打开：

```yaml
stability:
  fgm_feedback_max: 0.05
  logit_clamp: 30.0

train:
  skip_nonfinite: true
  max_consecutive_nonfinite: 3
  max_grad_norm: 1.0
```

含义：

- FGM cue feedback scale 会被 clamp 到 `[-0.05, 0.05]`
- 最终 logits 会 clamp 到 `[-30, 30]`
- loss 或梯度出现 NaN/Inf 时跳过 batch
- 连续 3 个非有限 batch 会停止训练，避免继续把模型跑坏
- checkpoint 保存前仍会检查 state_dict 是否含非有限 tensor

## 7. 消融配置生成

可以把 `ablation_presets` 展开成普通 yml：

```bash
python tools/materialize_ablation_configs.py \
  --config configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml \
  --out_dir rgfgm_hp3d_fpm_ablation
```

输出位置：

```text
configs/experiments/rgfgm_hp3d_fpm_ablation/
```

重点建议先跑：

- `A2_reliability_gated_fgm`
- `A3_rgfgm_plus_hp3d_noise`
- `A4_rgfgm_plus_prototype_memory`
- `A5_full_rgfgm_hp3d_fpm`
- `A7_full_alpha001`
- `A10_gate_bias_minus3`

## 8. 测试选择标准

测试 summary 已增加：

```text
same_source_avg IoU
cross_source_opn IoU
pareto_score = same_source_avg + 0.7 * OPN
```

不要只看 DVI/CPNET 同源提升。推荐选择规则：

```text
same_source_avg >= 当前稳定 FGM baseline
OPN_20 不低于 CCM-only 或稳定 FGM 参考
pareto_score 提升
reliability_gate_mean_OPN 理想情况下低于 DVI/CPNET
forensic_adapter_alpha <= alpha_max
```

## 9. 与旧版本的关系

旧配置仍保留：

```text
configs/b23_ccm_lite_lora32.yml
configs/b23_ccm_fgm_lite_lora32.yml
configs/b23_ccm_fgm_forensic_gated_lora32.yml
```

新版本只新增配置和模块，不删除旧 FGM raw cue bank 路径。旧配置没有 `reliability_gate / forensic_adapter / prototype_memory` 字段时，这些模块参数会被冻结，不参与训练。
