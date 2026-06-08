# DINOv3 B23 TFCU CCM/FGM Lite

这是一个独立的新项目，用于验证：

```text
DINOv3 ViT-L/16 B23 native 32x32 feature
+ LoRA rank 32
+ CCM-Lite clip 内短时 anomaly cue correlation
+ FGM-Lite historical forgery cue bank propagation
+ Low-res concat fusion
+ Lite Boundary Decoder
```

它不使用旧项目的 highres DPT pyramid、P2/P3/P4/P5 FPN、RGB detail stem 或 512 多通道 decoder。512 阶段只对 1-channel mask logit 做双线性上采样，避免显存被高分辨率多通道特征吃满。

## 模型结构

输入为 `[B,M,K,3,512,512]`，默认 `M=4,K=4`。DINOv3 ViT-L/16 的第 23 个 block 输出 B23 patch feature：

```text
[B*M*K, 1024, 32, 32]
```

每个 clip 依次执行：

```text
B23 x_m [B,K,1024,32,32]
  -> CCM-Lite masked cross-attention
  -> FGM-Lite cue bank propagation + aggregation
  -> concat low-res fusion [B*K,128,32,32]
  -> LiteBoundaryDecoder 32->64->128->256
  -> final 1-channel logit upsample to 512
```

## CCM-Lite

CCM 使用 clip 内 frame-wise masked cross-attention。Q 是 32x32 full tokens，K/V 先池化到 16x16 tokens，并使用 lower-triangular causal frame mask。训练时可启用 random attention mask，保留概率默认 0.7。融合方式是 residual concat，`alpha_cc` 初始为 0.002，最后 1x1 fuse conv 零初始化，初始近似 identity。

## FGM-Lite

FGM 的 bank 存储聚合后的 anomaly cue，而不是完整 B23 feature：

```text
C_m: [B,64,16,16]
```

传播阶段用当前 `f_cc_m` 的 32x32 tokens 作为 Q，历史 bank cue 的 16x16 tokens 作为 K/V。聚合阶段使用中心帧 cue 加 first-last diff，再与 propagated cue 做 cross-attention。feature-shift prompt 使用 `current_key - hist_cue` 生成 prompt 后加回历史 cue。

## 整视频 FGM Bank 训练

`configs/b23_ccm_fgm_lite_lora32.yml` 默认使用方案 2 的训练方式：

```yaml
fgm_bank:
  train_full_video_windows: true
  stateful_train: true
  stateful_eval: true
  reset_on_new_video: true
  detach_cross_window: true

val_full_video: true
test_full_video: true
test_max_clips: 2
```

数据集会在初始化时为每个视频提前生成完整的、按时间顺序排列的 window 计划。训练时可以 shuffle 视频顺序，但同一个视频内部的 window 不会 shuffle，因此 FGM bank 可以从 window 0 一直延续到该视频最后一个 window。跨到新视频时 bank 会自动 reset。

注意：提前写好的是整视频 window/bank 运行计划，不是提前缓存 cue 特征值。cue 由当前模型在线产生，随着 LoRA、CCM、FGM、decoder 权重变化而变化；提前缓存 cue 会让 bank 与当前模型不一致。当前配置通过 `detach_cross_window: true` 只跨 window 传递记忆值，不跨整段视频反传计算图，显存更稳。

DDP 下 sampler 会把完整视频分配给 rank，并在训练时补齐各 rank 的 step 数，避免某个 rank 提前结束。因为每个 batch 都拥有一个明确的视频状态，整视频 bank 训练要求：

```yaml
batch_size: 1
```

训练日志中会出现：

```text
video=... window=... bank=...
```

其中 `bank` 表示当前 FGM cue bank 已累计的历史 cue 数，最大受 `tfcu.fgm.bank_len` 控制。

## 时序增强

当前 FGM 配置打开了两类训练端时序鲁棒增强：

```yaml
temporal_augment:
  frame_swap:
    enabled: true
    prob: 0.20
    max_swaps: 1
    local_radius: 2
  frame_drop:
    enabled: true
    prob: 0.20
    max_drops: 1
```

`frame_swap` 在局部范围内交换相邻时间位置，模拟轻微时序错乱。`frame_drop` 是真丢帧：从当前 window 的有效时间序列中移除帧，后续帧左移，再从后续真实帧补齐；若已经到视频末尾，只重复最后帧并把对应位置标为 invalid，loss 和 metric 会通过 `valid_mask` 过滤这些 padding 位置。

## 训练

```bash
bash scripts/train_ddp.sh configs/b23_ccm_lite_lora32.yml
bash scripts/train_ddp.sh configs/b23_ccm_fgm_lite_lora32.yml
bash scripts/train_ddp.sh configs/b23_ccm_fgm_forensic_gated_lora32.yml
```

其中 `configs/b23_ccm_fgm_lite_lora32.yml` 是当前方案 2：整视频有状态 FGM bank + 时序增强。旧版随机 clip 训练配置保留为：

```bash
bash scripts/train_ddp.sh configs/b23_ccm_fgm_lite_lora32_legacy_random_clip.yml
```

旧版配置会保存到 `runs/b23_ccm_fgm_lite_lora32_legacy_random_clip`，不会覆盖当前方案 2 的 `runs/b23_ccm_fgm_lite_lora32`。

`configs/b23_ccm_fgm_forensic_gated_lora32.yml` 是泛化版：保留 DINOv3 B23 + CCM + FGM 的时序优势，同时加入低层 residual/noise forensic branch 和 quality-gated FGM bank，输出目录为：

```text
runs/b23_ccm_fgm_forensic_gated_lora32
```

详细说明见：

```text
DINOv3_B23_Forensic_Gated_FGM_Guide.md
```

如果只想单卡调试，把配置里的：

```yaml
ddp:
  auto_torchrun: false
```

## 测试

```bash
bash scripts/test_full_video.sh runs/b23_ccm_fgm_lite_lora32/best_iou.pt
```

测试会分别输出 DVI、CPNET、OPN 指标。

## 消融

```bash
bash scripts/ablate_disable_fgm.sh runs/b23_ccm_fgm_lite_lora32/best_iou.pt
bash scripts/ablate_shuffle_bank.sh runs/b23_ccm_fgm_lite_lora32/best_iou.pt
python test.py --checkpoint runs/b23_ccm_fgm_lite_lora32/best_iou.pt --zero-bank
python test.py --checkpoint runs/b23_ccm_fgm_lite_lora32/best_iou.pt --disable-ccm
```

配置文件在 `configs/ablations/` 下，包括 static、CCM-only、no-prompt、feature-prompt、shuffle-bank、zero-bank-test、param-matched spatial adapter。

## 常见 Debug Shape

```text
input_video_shape: (B,4,4,3,512,512)
b23_feature_shape: (B*16,1024,32,32)
ccm_q_shape: (B,4096,128)
ccm_kv_shape: (B,1024,128)
fgm_cue_shape: (B,64,16,16)
fusion_out_shape: (B*K,128,32,32)
decoder_f64_shape: (B*K,96,64,64)
decoder_f128_shape: (B*K,48,128,128)
decoder_f256_shape: (B*K,16,256,256)
decoder_logit512_shape: (B*K,1,512,512)
```

## 独立性

新项目不从旧项目 import，也不使用软链接。需要把本地 DINOv3 repo 和权重放到：

```text
./dinov3
./dinov3/dinov3_weight/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```
