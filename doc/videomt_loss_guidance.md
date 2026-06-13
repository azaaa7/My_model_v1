# VidEoMT-style Window Query Fusion 的 Loss 设计与代码修改指南

## 0. 修改目标

当前新结构目标是：

```text
DINOv3-B23
+ Window input [B, W, T, 3, H, W]
+ VidEoMT-style query propagation / query fusion
+ LiteBoundaryDecoder 或 query-mask decoder
- CCM
- FGM
- FGM bank
- 多个复杂辅助 loss
```

Loss 设计也应同步简化。不要再保留过多辅助监督，例如 CCM mask loss、FGM cue loss、temporal cue loss、quality gate loss 等。  
本版本建议将 loss 控制在 **2 个主 loss + 1 个可选边界 loss** 内。

---

## 1. 从论文学习到的 Loss 原则

### 1.1 VidEoMT 的启发

VidEoMT 没有给 query propagation / query fusion 单独设计复杂监督。  
它采用的是与 Mask2Former 类似的目标：

```text
classification cross-entropy
+ binary cross-entropy for masks
+ Dice loss for masks
```

VidEoMT 的重点不是添加很多 loss，而是通过 **query propagation + query fusion** 让 query 自己在主任务监督下学习时序一致性。

迁移到本项目时，video inpainting 检测是二值 mask localization，不需要分类 CE。  
因此应简化为：

```text
BCEWithLogits loss
+ Dice loss
```

这就是主 loss。

---

### 1.2 RelayFormer 的启发

RelayFormer 用于 visual manipulation localization，它的 loss 更接近本项目任务。  
它使用：

```text
BCE loss
+ edge loss
```

其中 edge loss 也是 BCE 的形式，只是在边界区域上计算，用来强调篡改边界。

这说明 manipulation localization 不需要很多复杂监督。  
本项目可以保留一个轻量边界 loss，但必须作为可选项，不应成为核心依赖。

---

## 2. 最终推荐 Loss

第一版推荐：

```text
L = L_mask + λ_dice * L_dice + λ_edge * L_edge
```

其中：

```text
L_mask = BCEWithLogits(pred_logits, gt_mask)
L_dice = DiceLoss(sigmoid(pred_logits), gt_mask)
L_edge = BCEWithLogits(pred_edge_logits, gt_edge_mask)
```

推荐默认权重：

```yaml
loss:
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.2
```

如果 decoder 暂时没有 edge head，则使用：

```yaml
loss:
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.0
```

最终训练目标应保持简单：

```text
主版本：BCE + Dice
增强版本：BCE + Dice + Edge
```

不要超过三个 loss。

---

## 3. 不建议保留的 Loss

新模型不应再使用以下 loss：

```text
ccm_mask_loss
fgm_mask_loss
fgm_cue_loss
temporal_cue_loss
bank_consistency_loss
contrastive_reid_loss
query_consistency_loss
quality_gate_loss
motion_loss
optical_flow_loss
```

原因：

1. 新模型已经删除 CCM / FGM，不应继续监督不存在的中间模块。
2. VidEoMT 的关键是让 query 在主任务监督下学习传播，而不是给传播 query 额外加监督。
3. video inpainting 检测的数据标注通常只有二值 mask，额外伪标签容易引入噪声。
4. loss 太多会让论文叙事变弱：无法证明简单 query fusion 本身有效。

---

## 4. Ground Truth 形状规范

模型输出：

```python
logits: [B, W, T, 1, H, W]
```

ground truth mask 建议统一为：

```python
masks: [B, W, T, 1, H, W]
```

如果 dataloader 输出为：

```python
[B, W, T, H, W]
```

则在 loss 内部转换：

```python
if masks.ndim == 5:
    masks = masks.unsqueeze(3)
```

mask 应为 float：

```python
masks = masks.float()
```

取值范围：

```text
0 = authentic / background
1 = manipulated / inpainted
```

---

## 5. BCEWithLogits Loss

不要先 sigmoid 再 BCE。  
应使用：

```python
F.binary_cross_entropy_with_logits(logits, masks)
```

原因：

```text
BCEWithLogits = sigmoid + BCE 的数值稳定版本
```

实现：

```python
def bce_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets.float())
```

如果正负样本严重不均衡，可以可选加入 `pos_weight`，但第一版不建议默认启用。

可选配置：

```yaml
loss:
  use_pos_weight: false
  pos_weight: 3.0
```

对应实现：

```python
pos_weight = torch.tensor([cfg.loss.pos_weight], device=logits.device)
loss = F.binary_cross_entropy_with_logits(
    logits,
    targets,
    pos_weight=pos_weight,
)
```

---

## 6. Dice Loss

Dice loss 用来解决前景区域小、正负样本不均衡的问题。  
对 inpainting localization 很重要，因为篡改区域可能很小。

推荐实现：

```python
def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = tuple(range(2, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
```

对于形状 `[B, W, T, 1, H, W]`，上面会在 `T/1/H/W` 维度上聚合。  
如果希望每帧独立计算，也可以 reshape：

```python
logits = logits.reshape(-1, 1, H, W)
targets = targets.reshape(-1, 1, H, W)
```

第一版推荐每帧独立 Dice，更稳定：

```python
B, Wn, T, C, H, W_img = logits.shape
logits_2d = logits.reshape(B * Wn * T, C, H, W_img)
targets_2d = targets.reshape(B * Wn * T, C, H, W_img)
loss_dice = dice_loss_from_logits(logits_2d, targets_2d)
```

---

## 7. 可选 Edge Loss

RelayFormer 的 edge loss 思想适合篡改检测，但不要让它复杂化。  
如果当前 decoder 已有 boundary head，可以使用：

```python
edge_logits = aux["boundary128"] 或 aux["edge_logits"]
```

如果没有 edge head，则跳过 edge loss。

### 7.1 生成 edge mask

建议使用 max-pool 形态学边界，避免依赖 OpenCV：

```python
def mask_to_edge(mask, kernel_size=5):
    """
    mask: [N, 1, H, W], float in {0,1}
    """
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0, 1)
    return edge
```

### 7.2 对齐 edge logits 和 edge target

如果 edge head 输出较低分辨率，例如 `[N,1,128,128]`，则：

```python
edge_target = F.interpolate(
    mask,
    size=edge_logits.shape[-2:],
    mode="nearest",
)
edge_target = mask_to_edge(edge_target)
```

然后计算：

```python
loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_target)
```

---

## 8. 新 Loss 类建议

新增文件：

```text
src/losses/videomt_loss.py
```

推荐实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def mask_to_edge(mask, kernel_size=5):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class VideoMTLoss(nn.Module):
    """
    Simple loss for VidEoMT-style video inpainting localization.

    Total:
        L = bce_weight * BCEWithLogits
          + dice_weight * Dice
          + edge_weight * EdgeBCE optional
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        edge_weight=0.0,
        edge_kernel_size=5,
        use_pos_weight=False,
        pos_weight=1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_value = float(pos_weight)

    def forward(self, outputs, targets):
        logits = outputs["logits"]

        if isinstance(targets, dict):
            masks = targets.get("masks", targets.get("mask"))
        else:
            masks = targets

        if masks is None:
            raise ValueError("VideoMTLoss requires target masks.")

        if masks.ndim == 5:
            masks = masks.unsqueeze(3)

        masks = masks.float().to(device=logits.device)

        if masks.shape[-2:] != logits.shape[-2:]:
            original_shape = masks.shape
            masks = masks.reshape(-1, 1, original_shape[-2], original_shape[-1])
            masks = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
            masks = masks.reshape(*logits.shape)

        logits_2d = logits.reshape(-1, 1, logits.shape[-2], logits.shape[-1])
        masks_2d = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1])

        loss_dict = {}

        if self.use_pos_weight:
            pos_weight = torch.tensor(
                [self.pos_weight_value],
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            pos_weight = None

        loss_bce = F.binary_cross_entropy_with_logits(
            logits_2d,
            masks_2d,
            pos_weight=pos_weight,
        )
        loss_dict["loss_bce"] = loss_bce

        loss_dice = dice_loss_from_logits(logits_2d, masks_2d)
        loss_dict["loss_dice"] = loss_dice

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        edge_logits = None
        aux = outputs.get("aux", {})
        if isinstance(aux, dict):
            edge_logits = aux.get("edge_logits", None)
            if edge_logits is None:
                edge_logits = aux.get("boundary128", None)

        if self.edge_weight > 0 and edge_logits is not None:
            edge_logits_2d = edge_logits.reshape(
                -1, 1, edge_logits.shape[-2], edge_logits.shape[-1]
            )

            edge_target = F.interpolate(
                masks_2d,
                size=edge_logits_2d.shape[-2:],
                mode="nearest",
            )
            edge_target = mask_to_edge(
                edge_target,
                kernel_size=self.edge_kernel_size,
            )

            loss_edge = F.binary_cross_entropy_with_logits(
                edge_logits_2d,
                edge_target,
            )
            loss_dict["loss_edge"] = loss_edge
            total = total + self.edge_weight * loss_edge
        else:
            loss_dict["loss_edge"] = logits_2d.new_tensor(0.0)

        loss_dict["loss_total"] = total
        return total, loss_dict
```

---

## 9. 配置修改

新增或替换 loss 配置：

```yaml
loss:
  name: VideoMTLoss
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.2
  edge_kernel_size: 5
  use_pos_weight: false
  pos_weight: 1.0
```

如果第一版 decoder 没有稳定的 boundary 输出：

```yaml
loss:
  name: VideoMTLoss
  bce_weight: 1.0
  dice_weight: 1.0
  edge_weight: 0.0
```

---

## 10. Trainer 修改建议

在 loss 构建处加入：

```python
from src.losses.videomt_loss import VideoMTLoss


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "VideoMTLoss")

    if name == "VideoMTLoss":
        return VideoMTLoss(
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            edge_weight=loss_cfg.get("edge_weight", 0.0),
            edge_kernel_size=loss_cfg.get("edge_kernel_size", 5),
            use_pos_weight=loss_cfg.get("use_pos_weight", False),
            pos_weight=loss_cfg.get("pos_weight", 1.0),
        )

    raise ValueError(f"Unknown loss: {name}")
```

训练 loop 中：

```python
outputs = model(images)
loss, loss_dict = criterion(outputs, masks)
```

日志只记录：

```text
loss_total
loss_bce
loss_dice
loss_edge
```

不要再记录大量 CCM/FGM auxiliary losses。

---

## 11. Aux 字段兼容

新模型的 aux 建议包含：

```python
aux = {
    "videomt_queries": query_states,
    "edge_logits": edge_logits,       # optional
    "boundary128": boundary128,       # optional for old decoder compatibility
    "ccm_mask32": None,
    "fgm_mask32": None,
    "fgm_cue": None,
}
```

Loss 中只使用：

```text
outputs["logits"]
outputs["aux"]["edge_logits"] 或 outputs["aux"]["boundary128"]
```

不使用 query state 计算 loss。  
Query state 只用于 debug / visualization。

---

## 12. 为什么不对 query 加 loss

不要给 `videomt_queries` 加 consistency loss。  
原因：

1. VidEoMT 本身没有对传播 query 单独加监督。
2. Query 的功能应通过 mask loss 端到端学习。
3. video inpainting 的 query 不对应明确实例 ID，强行约束 query 一致性可能会限制模型学习。
4. 双向传播已经提供了 temporal context，不需要额外 temporal smoothness loss。

如需稳定训练，优先使用：

```text
residual alpha init = 0
learning rate warmup
lower lr for DINOv3
higher lr for query fusion / decoder
```

不要通过增加 loss 稳定训练。

---

## 13. 推荐消融

Loss 消融只做三个：

```text
L1: BCE
L2: BCE + Dice
L3: BCE + Dice + Edge
```

不要做十几个 loss 组合。

预期：

```text
BCE:
  收敛稳定，但小区域可能漏检。

BCE + Dice:
  主推荐，改善小目标和前景不均衡。

BCE + Dice + Edge:
  可能改善边界，但如果边界标注粗糙，收益有限。
```

最终论文或实验报告中主模型使用：

```text
BCE + Dice
```

增强版附加：

```text
+ Edge loss
```

---

## 14. 完成标准

完成后应满足：

```text
1. 新 loss 文件可以正常 import。
2. VideoMTLoss 只依赖 outputs["logits"] 和可选 edge logits。
3. 默认只有 BCE + Dice 两个主 loss。
4. edge loss 可通过 edge_weight=0 关闭。
5. 不再计算 CCM / FGM / cue / bank / contrastive / query consistency loss。
6. 训练日志简洁，只包含 total、bce、dice、edge。
7. 新 loss 能处理 [B,W,T,1,H,W] 输出。
8. 新 loss 能处理 mask 为 [B,W,T,H,W] 或 [B,W,T,1,H,W]。
```
