# DINOv3 B23 Forensic-Gated FGM 泛化版说明

## 版本定位

新增配置：

```bash
configs/b23_ccm_fgm_forensic_gated_lora32.yml
```

该版本保留原有 DINOv3 B23 + LoRA32 + CCM + FGM 的高精度路线，同时新增两类面向泛化的约束：

```text
DINOv3 B23 high-level semantic feature
+ CCM clip 内短时相关
+ FGM 跨 window 历史 cue propagation
+ low-level residual/noise forensic branch
+ quality-gated FGM cue bank
+ Lite Boundary Decoder
```

原配置不会被覆盖：

```bash
configs/b23_ccm_fgm_lite_lora32.yml
configs/b23_ccm_fgm_lite_lora32_legacy_random_clip.yml
configs/b23_ccm_lite_lora32.yml
```

新版本训练输出目录为：

```bash
runs/b23_ccm_fgm_forensic_gated_lora32
```

## 为什么要加这个版本

当前 FGM 版本在 DVI/CPNET 同源测试上更强，但跨域泛化提升有限。主要原因是 FGM bank 记的是 DINO B23 高层 cue，容易把训练源里的背景、压缩、inpainting 风格一起记进去。

新版本把证据拆成两条：

```text
高层语义/上下文：DINOv3 B23 + CCM + FGM
低层取证痕迹：Residual/Noise Forensic Branch
```

同时，FGM bank 不再无条件写入每个 cue，而是根据当前预测质量筛选，减少低质量历史 cue 污染后续窗口。

## 新增 1：Residual/Noise Forensic Branch

配置：

```yaml
forensic_branch:
  enabled: true
  out_channels: 32
  hidden_channels: 32
  fusion_channels: 128
  target_resolution: 32
  alpha_init: 0.01
  zero_init_fusion: true
  detach_input: false
```

输入仍是原始 RGB 帧 `[B*M*K,3,512,512]`。分支先转换为灰度图，然后提取：

```text
gray
laplacian high-pass
sobel-x
sobel-y
local residual = gray - local mean
gradient magnitude
abs high-pass residual
```

这些特征被池化到 32x32，并用轻量卷积投影到 128 通道，加到 temporal fusion 输出上：

```text
f32 = temporal_fusion(B23/CCM/FGM) + alpha * forensic_feature
```

`zero_init_fusion: true` 会让该分支初始接近 0，不会一开始破坏已有模型行为；训练后由 `forensic_branch.alpha` 学习它应该贡献多少。

## 新增 2：Quality-Gated FGM Bank

配置：

```yaml
fgm_bank:
  quality_gate:
    enabled: true
    warmup_items: 1
    keep_threshold: 0.20
    scale_min: 0.25
    area_min: 0.001
    area_max: 0.70
    min_similarity: 0.05
    confidence_weight: 0.45
    area_weight: 0.25
    consistency_weight: 0.30
```

FGM 仍然先读取历史 bank 做当前窗口 propagation；区别在于当前窗口结束后，新的 cue 入库前会计算质量分数：

```text
quality = confidence + area sanity + cue consistency
```

其中：

```text
confidence：预测概率熵越低，说明模型越确定
area sanity：mask 面积不能太小或太大
cue consistency：当前 cue 与上一条 bank cue 不能完全不相关
```

如果质量低于阈值，该 cue 不进入 bank；如果质量中等，会按质量缩放后再入库。这样可以减少错误记忆在长视频中被连续传播。

## 关键参数选择

相比 `b23_ccm_fgm_lite_lora32.yml` 的 more 版本，新配置更保守：

```yaml
tfcu:
  ccm:
    alpha_init: 0.002
  fgm:
    bank_len: 3
    aggregation:
      diff_scale: 0.75
      feedback_init: 0.005
    propagation:
      topk: 96

aux_loss:
  fgm_mask32: {enabled: true, weight: 0.03}

optimizer:
  lr_ccm: 5.0e-5
  lr_fgm: 5.0e-5
  lr_forensic: 5.0e-5
```

目的：

```text
降低 CCM/FGM 对源域时序风格的过强拟合
保留 FGM 对真实视频连续性的帮助
让低层取证分支补充跨域稳定痕迹
```

## 训练命令

```bash
cd /home/wzk/Exp/dinov3_b23_tfcu_ccm_fgm_lite
bash scripts/train_ddp.sh configs/b23_ccm_fgm_forensic_gated_lora32.yml
```

如果从一个干净的 FGM checkpoint 初始化，把路径换成你实际要继承的 `best_iou.pt`：

```bash
cd /home/wzk/Exp/dinov3_b23_tfcu_ccm_fgm_lite
bash scripts/train_ddp.sh configs/b23_ccm_fgm_forensic_gated_lora32.yml \
  --checkpoint runs/b23_ccm_fgm_lite_lora32/best_iou.pt
```

也可以使用 `runs/b23_ccm_fgm_lite_lora32_more/best_iou.pt`，前提是它是干净的 best checkpoint。不要从已经 NaN 污染的 `latest.pt` 恢复。

## 测试命令

```bash
cd /home/wzk/Exp/dinov3_b23_tfcu_ccm_fgm_lite
bash scripts/test_full_video.sh runs/b23_ccm_fgm_forensic_gated_lora32/best_iou.pt
```

测试会分别输出 DVI、CPNET、OPN。泛化选择建议不要只看 DVI/CPNET 均值，应同时看 OPN：

```text
source_mean = mean(DVI, CPNET)
general_score = harmonic_mean(source_mean, OPN)
```

如果目标是未知数据集，优先选择 `general_score` 或 `min(DVI, CPNET, OPN)` 更高的 checkpoint。

## Debug 观察点

训练日志/调试信息中建议关注：

```text
forensic_alpha
forensic_norm
fgm_bank_quality
fgm_bank_quality_keep
fgm_bank_quality_confidence
fgm_bank_quality_area
fgm_bank_quality_consistency
ccm_alpha
fgm_cue_feedback_scale
```

合理现象：

```text
forensic_alpha 从 0.01 缓慢变化，不应突然爆大
fgm_bank_quality_keep 不是一直 0，也不是所有低质量窗口都无脑 1
ccm_alpha / fgm_cue_feedback_scale 温和增长
OPN 不应明显掉，而 DVI/CPNET 尽量保持
```

## 何时继续训练或停止

如果出现下面情况，建议优先使用 best checkpoint，不继续追 latest：

```text
DVI/CPNET 继续涨，但 OPN 连续下降
train iou 继续涨，val mean_iou 长时间横盘
forensic_alpha 或 FGM gate 突然异常变大
出现 nonfinite skip 或 loss nan
```

这个版本的目标不是单点同源最高，而是让高精度和跨域泛化同时更稳。
