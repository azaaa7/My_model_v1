# B23 VidEoMT Window Query Fusion 最终版说明

## 1. 目标

本版本实现 `agent_videomt_final_implementation.md` 中要求的最终版结构：

```text
DINOv3-B23
+ LoRA rank=32
+ Learned forgery queries
+ VidEoMT-style window query propagation
+ forward/backward 双向平均
+ LiteBoundaryDecoder
+ BCE + Dice loss
```

它不是在旧 CCM/FGM 上继续堆模块，而是用窗口内 query propagation 替代旧的 CCM、FGM 和 FGM bank。

## 2. 模型结构

新增模型：

```text
src/models/b23_videomt_window_model.py
class B23VideoMTWindowModel
```

输入：

```text
[B, W, T, 3, H, W]
```

兼容旧输入：

```text
[B, T, 3, H, W] -> 自动扩展为 [B, 1, T, 3, H, W]
```

输出：

```python
{
    "logits": Tensor[B, W, T, 1, H, W],
    "aux": {
        "videomt_queries": Tensor[B, W, T, Nq, C],
        "edge_logits": optional Tensor[B, W, T, 1, 128, 128],
        "boundary128": optional Tensor[B, W, T, 1, 128, 128],
    }
}
```

主路径：

```text
video window
  -> DINOv3B23Encoder
  -> WindowQueryFusion
  -> 1x1 feature_proj: 1024 -> 128
  -> LiteBoundaryDecoder
  -> logits 512x512
```

## 3. WindowQueryFusion

新增文件：

```text
src/models/videomt/window_query_fusion.py
```

核心公式：

```text
Q_f_in[0] = Q_lrn
Q_f_in[t] = Linear_f(Q_f_out[t-1]) + Q_lrn

Q_b_in[T-1] = Q_lrn
Q_b_in[t] = Linear_b(Q_b_out[t+1]) + Q_lrn

Q_out[t] = 0.5 * (Q_f_out[t] + Q_b_out[t])
```

注意：

```text
没有 CCM
没有 FGM
没有 FGM bank
没有 gate_f / gate_b
没有 sigmoid MLP gate
没有 query consistency loss
```

`residual_alpha` 只是把 query context 加回 B23 feature 的残差缩放，默认初始值为 `0.0`，用于稳定训练，不是质量门控。

## 4. Loss

新增 loss：

```text
src/losses/videomt_loss.py
class VideoMTLoss
```

默认监督目标：

```text
L = 1.0 * BCEWithLogits + 1.0 * Dice
```

Edge BCE 已实现但默认关闭：

```yaml
edge_weight: 0.0
```

最终配置中 `decoder.boundary_head.enabled=false`。如果后续要启用 Edge BCE，需要同时把：

```yaml
loss.edge_weight: 0.2
decoder.boundary_head.enabled: true
```

否则 DDP 训练时 boundary head 会产生未使用参数。

同理，最终版默认只监督最终 `logits`，因此 `decoder.mask128_head.enabled=false`。如果后续要恢复 `mask128` deep supervision，必须同时启用对应 aux loss；否则 `mask_head128` 的参数也会触发 DDP unused parameter。

该 loss 不使用 token loss、temporal smoothness、CCM/FGM aux loss 或 bank consistency。

## 5. 配置文件

新增最终版配置：

```text
configs/b23_videomt_window.yaml
```

关键字段：

```yaml
model:
  name: "B23VideoMTWindowModel"

videomt:
  enabled: true
  num_queries: 16
  bidirectional: true
  residual_alpha_init: 0.0

tfcu:
  version: "videomt_window"
  ccm:
    enabled: false
  fgm:
    enabled: false

fgm_bank:
  stateful_train: false
  stateful_eval: false

loss:
  type: "videomt"
  name: "VideoMTLoss"
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.0
```

## 6. 训练和测试命令

训练：

```bash
bash scripts/train_ddp.sh configs/b23_videomt_window.yaml
```

从 latest 恢复：

```bash
bash scripts/train_ddp.sh configs/b23_videomt_window.yaml --checkpoint runs/b23_videomt_window/latest.pt
```

测试：

```bash
bash scripts/test_full_video.sh runs/b23_videomt_window/best_iou.pt configs/b23_videomt_window.yaml
```

## 7. 日志检查点

训练日志中应能看到：

```text
loss_total
loss_bce
loss_dice
videomt_query_alpha
```

如果 `videomt_query_alpha` 长期为 0，说明 query context 没有被实际注入到 feature；如果它逐渐偏离 0，说明 VidEoMT 分支已经参与训练。

旧 FGM bank 相关日志在此配置下不会出现，训练代码会自动检测模型是否支持 `new_fgm_bank`，新模型不支持时不会传入 `fgm_bank` / `return_fgm_bank`。

## 8. 与旧版本的区别

| 项目 | 旧 CCM/FGM 版本 | VidEoMT 最终版 |
|---|---|---|
| 短时模块 | CCM cross-frame cue | learned query 与 patch token 交互 |
| 长时模块 | FGM bank | 无 bank |
| 跨 window 状态 | 可选 stateful bank | 无历史状态 |
| 门控 | 多种 gate / quality gate | 无 gate |
| loss | 多 loss/aux 可叠加 | BCE + Dice |
| 训练稳定性 | 依赖 bank/aux 配置 | 更简单，调参面更小 |

本版本适合用来验证：不依赖显式 memory bank 的 VidEoMT-style query propagation 是否能在视频 inpainting 检测中提供更好的泛化稳定性。
