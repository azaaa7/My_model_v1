# b23_videomt_window_final 全量修复指导文档

> 目标：把 `b23_videomt_window_final.yaml` 对应的训练路径修成真正可训练、可验证、可复现的 VidEoMT-like encoder-query-mask final model。
>
> 训练命令：
>
> ```bash
> bash scripts/train_ddp.sh configs/b23_videomt_window_final.yaml
> ```
>
> 验收命令：
>
> ```bash
> python tools/check_b23_videomt_final.py
> ```
>
> 最终结构必须是：
>
> ```text
> Linear(prev_q) + Q_lrn
>   -> query token 进入 DINOv3 最后 4 个 block
>   -> patch feature
>   -> decoder 导出 f128
>   -> query mask head 生成 query_logits/query_scores
>   -> query-level masks 聚合成最终 logits
>   -> VideoMTQueryMaskLoss 同时监督 final logits 和 query masks
> ```
>
> 禁止退回到旧结构：
>
> ```text
> query_states.mean(...)
>   -> query_to_feature
>   -> features + residual_alpha * query_context
>   -> decoder logits
> ```

---

## 0. 当前必须处理的核心问题

请不要只看配置文件是否正确。必须运行代码路径检查。当前项目中可能已经有部分文件被改过，但仍需要逐项确认以下事实：

1. `B23VideoMTWindowModel` 不能再调用 `WindowQueryFusion`。
2. `DINOv3B23Encoder.forward_with_queries()` 必须真实把 query token 放进 DINOv3 最后 4 个 block。
3. 最终 logits 必须来自 `QueryMaskHead` 的聚合结果，而不是 decoder 的 `mask256/logits`。
4. `aux` 必须包含：
   - `videomt_queries`
   - `query_logits`
   - `query_scores`
   - `edge_logits` 或 `boundary128`
5. `VideoMTQueryMaskLoss` 必须真实读取 `aux["query_logits"]` 并产生非零、有限的 query-level loss。
6. 训练、验证、测试都必须支持跨 window 的 `videomt_state["prev_q"]`。
7. optimizer 必须给 `query_controller` 和 `query_mask_head` 单独 param group。
8. `tools/check_b23_videomt_final.py` 必须 forward + loss + backward 全通过。
9. 训练日志中必须能看到：
   - `loss_query_bce`
   - `loss_query_dice`
   - `loss_query_no_object`
   - query/mask_head 梯度非空且有限。

---

## 1. 配置文件检查：`configs/b23_videomt_window_final.yaml`

必须确认以下配置存在且生效：

```yaml
model:
  name: "B23VideoMTWindowModel"
  version: "videomt_encoder_query_mask_final"

dinov3:
  output_block: 23
  query_injection:
    enabled: true
    inject_after_block: 19
    query_blocks: [20, 21, 22, 23]
    keep_cls_token: true
    patch_query_attention: true

videomt:
  enabled: true
  dim: 1024
  num_queries: 32
  bidirectional: false

  residual:
    enabled: false
    residual_alpha_init: 0.0

  propagation:
    enabled: true
    type: "linear_plus_learned"
    first_frame: "learned"
    prev_linear: true
    detach_within_window: false
    stateful_windows: true
    detach_across_windows: true
    carry_state_in_eval: true
    reset_state_per_video: true

  query_mask_head:
    enabled: true
    mask_dim: 256
    mask_feature_source: "decoder_128"
    mask_resolution: 128
    upsample_to_input: true
    aggregation:
      type: "logsumexp"
      temperature: 1.0
      use_query_score: true

decoder:
  export_features:
    enabled: true
    resolutions: [128]
  mask256_head:
    enabled: false

loss:
  type: "videomt_query_mask"
  name: "VideoMTQueryMaskLoss"
  query_mask:
    enabled: true
```

### 关键说明

- `mask256_head.enabled: false` 是合理的，但前提是最终 logits 已经由 `QueryMaskHead` 产生。
- 如果模型仍然使用 `dec_out["logits"]` 作为最终 logits，那么 `mask256_head.enabled: false` 会导致 logits 为 `None` 或训练失效。
- 不允许通过重新打开 `mask256_head` 来掩盖 query mask head 没有生效的问题。

---

## 2. 模型注册检查

### 文件：`src/models/__init__.py`

必须可以导入：

```python
from src.models import B23VideoMTWindowModel
```

`__all__` 中必须包含：

```python
"B23VideoMTWindowModel"
```

### 文件：`src/models/builder.py`

必须支持：

```python
def build_model(cfg):
    name = str((cfg.get("model", {}) or {}).get("name", ""))
    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)
```

### 验收

```bash
python - <<'PY'
from src.utils.config import load_config
from src.models.builder import build_model

cfg = load_config("configs/b23_videomt_window_final.yaml")
model = build_model(cfg)
print(type(model))
assert type(model).__name__ == "B23VideoMTWindowModel"
PY
```

---

## 3. 删除旧 `WindowQueryFusion` 默认路径

### 文件：`src/models/b23_videomt_window_model.py`

必须满足：

```python
from .videomt.query_encoder import VideoMTQueryController
from .videomt.query_mask_head import QueryMaskHead
```

不得出现：

```python
from .videomt import WindowQueryFusion
self.query_fusion = WindowQueryFusion(...)
self.query_to_feature(...)
query_states.mean(...)
residual_alpha
features + residual
```

### 必须存在的模块

```python
self.encoder = DINOv3B23Encoder(dinov3_cfg, lora_cfg)
self.query_controller = VideoMTQueryController(videomt_cfg)
self.feature_proj = ...
self.decoder = LiteBoundaryDecoder(decoder_cfg)
self.query_mask_head = QueryMaskHead(...)
```

### forward 必须接受

```python
def forward(
    self,
    video,
    mode=None,
    ablation=None,
    epoch=None,
    videomt_state=None,
    return_videomt_state=False,
    **kwargs,
):
```

保留 `**kwargs`，避免 trainer 仍传入旧参数时报错。

---

## 4. DINOv3 encoder：确认 query 真实进入最后 4 个 block

### 文件：`src/models/dinov3_b23_encoder.py`

必须有：

```python
def forward_with_queries(
    self,
    frames: torch.Tensor,
    query_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
```

### 必须满足的逻辑

1. 对 `frames` 做 DINO 标准 normalize。
2. 使用 DINOv3 原生 token 准备函数，例如：
   - `prepare_tokens_with_masks(frames)`
   - 或等价的 patch embed + cls/storage tokens + pos/rope 逻辑。
3. block `0..19` 只处理 native tokens。
4. block `20..23` 同时处理 native tokens + query tokens。
5. 输出：
   - `patch_features`: `[N, 1024, 32, 32]`
   - `out_queries`: `[N, 32, 1024]`

### query 插入位置要求

如果 DINOv3 block 使用 RoPE，并且 RoPE 默认只对 patch token 段做二维位置编码，则不要随意把 query 插在 patch token 后面。推荐：

```text
[cls/storage tokens] + [query tokens] + [patch tokens]
```

这样 patch tokens 仍在最后连续一段，便于 `_extract_patch_tokens()` 从末尾取 `32*32` 个 patch tokens。

### 必须保留 patch token 提取方式

```python
patch_tokens = native_or_all_tokens[:, -num_patches:, :]
```

### 验收

```bash
python - <<'PY'
import torch
from src.utils.config import load_config
from src.models.dinov3_b23_encoder import DINOv3B23Encoder

cfg = load_config("configs/b23_videomt_window_final.yaml")
enc = DINOv3B23Encoder(cfg["dinov3"], cfg["lora"]).cuda().train()
x = torch.rand(2, 3, 512, 512, device="cuda")
q = torch.randn(2, 32, 1024, device="cuda", requires_grad=True)
f, qo = enc.forward_with_queries(x, q)

print(f.shape, qo.shape)
assert f.shape == (2, 1024, 32, 32)
assert qo.shape == (2, 32, 1024)
loss = f.float().mean() + qo.float().mean()
loss.backward()
assert q.grad is not None
assert torch.isfinite(q.grad).all()
print("OK")
PY
```

---

## 5. query controller：确认 `Linear(prev_q)+Q_lrn`

### 文件：`src/models/videomt/query_encoder.py`

必须有：

```python
class VideoMTQueryController(nn.Module):
    def initial_queries(...)
    def make_input_queries(...)
```

核心公式必须是：

```python
if prev_q is None:
    return Q_lrn

if detach_prev:
    prev_q = prev_q.detach()

return self.prev_linear(prev_q.to(dtype=dtype)) + Q_lrn
```

### 禁止事项

- 不要做 bidirectional query。
- 不要做 query mean residual。
- 不要每个 window 都重新初始化 query。
- 不要跨 video 保留上一个 video 的 query state。

---

## 6. decoder：必须导出 `features128`

### 文件：`src/models/decoders/lite_boundary_decoder.py`

`forward()` 返回 dict 必须包含：

```python
{
    "features128": f128,
    "boundary128": boundary128,
    ...
}
```

### 验收

```python
dec_out = decoder(dec_in)
assert "features128" in dec_out
assert dec_out["features128"].shape[-2:] == (128, 128)
```

---

## 7. query mask head：最终 logits 必须来自它

### 文件：`src/models/videomt/query_mask_head.py`

必须输出：

```python
{
    "logits": logits,               # [B,W,T,1,512,512] 或 [B,T,1,512,512]
    "query_logits": query_logits,   # [B,W,T,Q,128,128]
    "query_scores": query_scores,   # [B,W,T,Q,1]
}
```

### 聚合建议

当前推荐：

```python
query_logits_for_agg = query_logits + query_scores[..., None]
logits = logsumexp(query_logits_for_agg / temp, dim=query_dim) * temp
```

建议加入或确认已经加入：

```python
normalize_logsumexp: true
```

等价实现：

```python
logits = logits - temp * math.log(max(1, num_queries))
```

原因：`logsumexp` 会天然随 query 数增加抬高背景 logits。32 个 query 时，不归一化会增加约 `log(32)=3.47` 的正偏置，容易导致 false positive。

### score head 初始化建议

建议把 query score head 最后一层 bias 初始化为负值，例如：

```python
nn.init.constant_(self.score_head[-1].bias, -2.0)
```

目的：训练早期降低 query 聚合导致的过分前景化。

---

## 8. `B23VideoMTWindowModel.forward()` 最终逻辑

每个 window 内逐帧处理：

```python
prev_q = videomt_state.get("prev_q") if videomt_state else None

for win_idx in range(num_windows):
    for frame_idx in range(num_frames):
        q_in = query_controller.make_input_queries(
            batch_size=B,
            device=device,
            dtype=frame.dtype,
            prev_q=prev_q,
            detach_prev=win_idx > 0 and frame_idx == 0 and detach_across_windows,
        )

        patch_feat, q_out = encoder.forward_with_queries(frame, q_in)
        prev_q = q_out.detach() if detach_within_window else q_out
```

decoder 和 query mask：

```python
dec_out = decoder(feature_proj(patch_features))
f128 = dec_out["features128"]
qmh_out = query_mask_head(query_states, f128, output_size=(H, W))
logits = qmh_out["logits"]
```

最终输出：

```python
out = {
    "logits": logits,
    "aux": {
        "videomt_queries": query_states,
        "query_logits": qmh_out["query_logits"],
        "query_scores": qmh_out["query_scores"],
        "edge_logits": edge_logits_or_none,
        "boundary128": edge_logits_or_none,
        "debug": debug,
    },
}
if return_videomt_state:
    out["videomt_state"] = {"prev_q": prev_q.detach()}
```

---

## 9. loss：必须真实监督 query masks

### 文件：`src/losses/videomt_query_mask_loss.py`

`VideoMTQueryMaskLoss` 必须兼容两种调用：

```python
criterion(out, target)
criterion(logits, target, aux=aux, epoch=epoch, include_aux=True)
```

### 必须返回 item keys

```python
{
    "loss_total": ...,
    "loss_bce": ...,
    "loss_dice": ...,
    "loss_query_bce": ...,
    "loss_query_dice": ...,
    "loss_query_no_object": ...,
    "loss_edge": ...,
    "main_loss": ...,
}
```

### query-level target 构造

稳定第一版使用 frame-level connected components：

1. 把 GT mask resize 到 query logits 分辨率，即 128。
2. 每帧做 connected components。
3. 最多保留 `Q=32` 个组件。
4. 用 Dice cost 做 Hungarian matching。
5. 匹配 query 监督对应 component。
6. 未匹配 query 监督全 0 mask，权重使用 `no_object.weight`。

### 必须避免的 bug

- `query_logits` 是 `[B,W,T,Q,128,128]`，不要误 reshape 成 `[BWT,1,H,W]`。
- targets 可能是：
  - `[B,W,T,1,512,512]`
  - `[B,T,1,512,512]`
  - `[B,1,512,512]`
- query loss 计算时必须统一 flatten 到 `[N,Q,Hq,Wq]`。
- 若某帧没有前景组件，所有 query 应进入 no-object loss，不能跳过该帧。
- loss item 必须是 Python float，不能保留 GPU tensor 导致日志序列化失败。

### 验收

训练第一个 epoch 内日志必须出现非零且有限的：

```text
loss_query_bce
loss_query_dice
loss_query_no_object
```

---

## 10. trainer：必须传递 VidEoMT state

### 文件：`src/train/trainer.py`

必须有：

```python
def _is_videomt_model(cfg):
    return cfg["model"]["name"] == "B23VideoMTWindowModel"

def _videomt_stateful_enabled(cfg, mode):
    prop = cfg.get("videomt", {}).get("propagation", {})
    if mode == "train":
        return bool(prop.get("stateful_windows", False))
    return bool(prop.get("carry_state_in_eval", True))
```

### 训练 loop

如果是 `B23VideoMTWindowModel`：

```python
out = model(
    images,
    mode="train",
    epoch=epoch,
    videomt_state=videomt_state,
    return_videomt_state=use_videomt_stateful,
)
if use_videomt_stateful:
    videomt_state = out.get("videomt_state", videomt_state)
```

不要传入旧的：

```python
fgm_bank
return_fgm_bank
```

即使传了，模型也必须通过 `**kwargs` 忽略。

### reset 规则

遇到以下情况必须 reset：

```python
video_id 改变
is_first_window == True
is_last_window == True 后下一个 batch
```

如果 batch 没有 `video_id/is_first_window/is_last_window`，则训练时不要强行 stateful，避免把不同视频串起来。

---

## 11. valid mask 过滤

### 文件：`src/train/trainer.py`

如果使用 `valid_mask`，必须能正确过滤以下 aux：

```python
query_logits:    [B,W,T,Q,H,W]
query_scores:    [B,W,T,Q,1]
videomt_queries: [B,W,T,Q,C]
edge_logits:     [B,W,T,1,H,W]
```

推荐逻辑：

```python
if tuple(x.shape[:valid_mask.ndim]) == tuple(valid_mask.shape):
    x = x.reshape(-1, *x.shape[valid_mask.ndim:])[valid]
```

不要把 query logits 错误当成普通 mask logits。

---

## 12. optimizer：必须包含 query 和 mask head 参数

### 文件：`src/train/optimizer.py`

必须有分组：

```python
lr_query = opt_cfg.get("lr_query", base_lr)
lr_mask_head = opt_cfg.get("lr_mask_head", base_lr)
```

参数分类：

```python
if "lora_" in name:
    group = "lora"
elif "decoder." in name or "feature_proj." in name:
    group = "decoder"
elif "query_controller." in name:
    group = "query"
elif "query_mask_head." in name:
    group = "mask_head"
else:
    group = "other"
```

### 验收

```bash
python - <<'PY'
from src.utils.config import load_config
from src.models.builder import build_model
from src.train.optimizer import build_optimizer

cfg = load_config("configs/b23_videomt_window_final.yaml")
model = build_model(cfg).cuda()
opt = build_optimizer(model, cfg)

names = [n for n,p in model.named_parameters() if p.requires_grad]
assert any("query_controller" in n for n in names)
assert any("query_mask_head" in n for n in names)
print("trainable query/mask params OK")
print([g["lr"] for g in opt.param_groups])
PY
```

---

## 13. scheduler：poly 必须生效

### 文件：`src/train/scheduler.py`

必须支持：

```yaml
scheduler:
  type: "poly"
```

实现必须考虑 param groups 原始 lr 不同：

```python
base_lrs = [group["lr"] for group in optimizer.param_groups]
```

不要把所有 param group 都压成同一个 lr。

---

## 14. eval/tester：必须传递 state

### 文件：`src/eval/tester.py`

如果 tester 单独构造模型，必须使用 `build_model(cfg)`，不要写死旧模型。

推理时：

```python
videomt_state = None
current_video_id = None

for batch in loader:
    if video_id != current_video_id or is_first_window:
        videomt_state = None
        current_video_id = video_id

    out = model(
        images,
        mode="eval",
        videomt_state=videomt_state,
        return_videomt_state=True,
    )
    videomt_state = out.get("videomt_state", videomt_state)

    if is_last_window:
        videomt_state = None
        current_video_id = None
```

---

## 15. 稳定性修复建议

### 15.1 `logsumexp` 背景偏置

建议在 `query_mask_head` 配置中新增：

```yaml
aggregation:
  type: "logsumexp"
  temperature: 1.0
  use_query_score: true
  normalize_logsumexp: true
```

并在代码中实现：

```python
if normalize_logsumexp:
    logits = logits - temp * math.log(max(1, q))
```

### 15.2 score head 初始偏置

建议：

```python
score_bias_init: -2.0
```

训练初期降低 false positive。

### 15.3 query loss warmup

如果训练前 10 个 epoch 不稳定，可加入：

```yaml
loss:
  query_mask:
    warmup_epochs: 5
```

实现：

```python
query_weight_scale = min(1.0, (epoch + 1) / warmup_epochs)
```

### 15.4 no-object 权重

初始建议：

```yaml
no_object:
  weight: 0.05
```

如果前景召回太低，再调回 `0.1`；如果 false positive 太高，调到 `0.2`。

### 15.5 LoRA 学习率

建议先使用：

```yaml
lr_lora: 2.0e-5
lr_query: 1.0e-4
lr_mask_head: 1.0e-4
lr_decoder: 1.0e-4
```

原因：query 进入 DINO 最后 4 层后，如果 LoRA 太快，容易破坏 DINOv3 的强静态表征。

---

## 16. 必须新增/更新验收脚本

### 文件：`tools/check_b23_videomt_final.py`

必须检查：

```python
out = model(x, mode="train", return_videomt_state=True)

assert out["logits"].shape == (1, 1, 5, 1, 512, 512)
assert out["aux"]["videomt_queries"].shape[:4] == (1, 1, 5, 32)
assert out["aux"]["query_logits"].shape[:4] == (1, 1, 5, 32)
assert out["aux"]["query_scores"].shape[:4] == (1, 1, 5, 32)
assert "videomt_state" in out
assert out["videomt_state"]["prev_q"].shape == (1, 32, 1024)

criterion = VideoMTQueryMaskLoss(cfg["loss"]).cuda()
loss, items = criterion(out, y)
loss.backward()

assert torch.isfinite(loss)
assert items["loss_query_bce"] > 0 or items["loss_query_dice"] > 0

has_query_grad = any(
    p.grad is not None and torch.isfinite(p.grad).all()
    for n, p in model.named_parameters()
    if "query_controller" in n or "query_mask_head" in n
)
assert has_query_grad
```

---

## 17. 训练前 1 小时排查清单

启动 DDP 前，先单卡跑：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_b23_videomt_final.py
```

然后跑 20 iteration debug train：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/b23_videomt_window_final.yaml --debug_steps 20
```

如果项目没有 `--debug_steps`，临时在 trainer 中限制前 20 个 step。

必须观察：

```text
logits finite: true
query_logits finite: true
query_scores finite: true
loss_total finite: true
loss_query_bce non-zero
loss_query_dice non-zero
query_controller grad finite
query_mask_head grad finite
LoRA grad finite, if enabled
```

---

## 18. 训练效果诊断表

| 现象 | 可能原因 | 修复 |
|---|---|---|
| loss_query_* 全是 0 | aux 没有 query_logits，或 loss 没读到 | 检查 model aux 和 VideoMTQueryMaskLoss |
| logits 全前景 | logsumexp 未归一化，score bias 太高 | 开启 normalize_logsumexp，score bias=-2 |
| logits 全背景 | no_object 太大，score bias 太低，query loss 过强 | no_object 降到 0.05，query loss warmup |
| 比 DINOv3-only 差很多 | query 进入 DINO 但扰乱 patch token，LoRA lr 太高 | 降低 lr_lora，冻结 LoRA 做 ablation |
| train 好 val 差 | query instance pseudo matching 过拟合 | 降 query loss 权重，增强数据，检查 OPN domain gap |
| eval full video 不稳定 | state 没 reset 或跨视频串联 | 检查 video_id/is_first_window/is_last_window |
| DDP 卡住 | find_unused_parameters=false 但有未使用分支 | 确认所有启用模块参与 loss，临时 true 定位 |

---

## 19. 必做 ablation

修复后至少跑以下 4 组，判断收益来自哪里：

### A. DINOv3-only baseline

```yaml
model:
  name: "B23TFCUCCMFGMLiteModel"  # 或项目当前 DINO-only 模型
```

记录 val IoU/F1。

### B. final model，但 LoRA 关闭

```yaml
lora:
  enabled: false
```

判断 query-mask 结构本身是否有效。

### C. final model，query state 不跨 window

```yaml
videomt:
  propagation:
    stateful_windows: false
    carry_state_in_eval: false
```

判断跨 window state 是否带来收益或污染。

### D. final model，全量

```yaml
lora.enabled: true
stateful_windows: true
carry_state_in_eval: true
```

目标是 D 至少不低于 A，并在 temporal consistency 和 full-video val 上超过 A。

---

## 20. 最终提交要求

提交前必须贴出以下日志或结果：

```text
1. tools/check_b23_videomt_final.py 完整输出
2. 单卡 20 step debug loss
3. optimizer param group 摘要
4. 第一个 epoch 的 train loss item
5. 第一次 val 的 IoU/F1/precision/recall
6. query_logits/query_scores 的 shape 和数值范围
```

必须确认：

```text
[OK] query token 进入 DINOv3 final blocks
[OK] final logits 来自 QueryMaskHead
[OK] query mask loss 非零且有限
[OK] query_controller/query_mask_head 有梯度
[OK] optimizer 包含 query/mask_head 参数
[OK] eval/test carry state 并按 video reset
[OK] 不再使用 WindowQueryFusion residual
```

---

## 21. 不要做的事

1. 不要把 `WindowQueryFusion` 重新接回最终路径。
2. 不要用 `mean(query)` 生成 feature residual。
3. 不要添加 gate 网络掩盖结构问题。
4. 不要引入 Mask2Former/DVIS/CAVIS tracker。
5. 不要只调学习率却不检查 query 是否进入 DINO block。
6. 不要只让 query mask head 做辅助输出；它必须产生最终 logits。
7. 不要让每个 window 都从 learned query 重新开始。
8. 不要在不同 video 之间传递 `prev_q`。
9. 不要用打开 `mask256_head` 的方式绕过 query-mask final logits。
10. 不要把 `query_logits` 在 valid mask 过滤中错误 reshape。

---

## 22. 建议的第一版稳定配置补丁

如果训练不稳定，先应用以下小补丁：

```yaml
videomt:
  query_mask_head:
    score_head:
      enabled: true
      hidden_dim: 256
      bias_init: -2.0
    aggregation:
      type: "logsumexp"
      temperature: 1.0
      use_query_score: true
      normalize_logsumexp: true

loss:
  query_mask:
    bce_weight: 0.3
    dice_weight: 0.3
    warmup_epochs: 5
    no_object:
      enabled: true
      weight: 0.05

optimizer:
  lr_lora: 2.0e-5
  lr_decoder: 1.0e-4
  lr_query: 1.0e-4
  lr_mask_head: 1.0e-4
```

如果验证集召回明显不足，再逐步把 query loss 恢复到：

```yaml
query_mask:
  bce_weight: 0.5
  dice_weight: 0.5
  no_object:
    weight: 0.1
```

---

## 23. 预期结果

修复前，模型名和配置虽然叫 final，但如果 query 没有真实进入 DINO block、query mask head 没有产生最终 logits、query loss 没有监督到 query logits，那么训练效果差是正常现象。

修复后预期：

1. 稳定性：loss 不应再出现 query 分支空转，`loss_query_bce/loss_query_dice` 应稳定非零。
2. 收敛速度：前 10~30 epoch 可能仍慢于 DINO-only，因为 query mask 和 pseudo component matching 需要适应。
3. in-domain val：通常应至少追平 DINOv3-only，并有机会提升约 1~4 个 IoU 点。
4. full-video/temporal：跨 window state 生效后，闪烁和断裂应明显减少，F1/IoU 可能小幅提升，但更明显的是连续帧稳定性。
5. out-domain OPN：不一定稳定提升，可能持平或小幅提升 0~2 个点；若训练集没有 OPN 风格，domain gap 仍是主要瓶颈。
6. 如果 query matching、logsumexp 归一化、no-object 权重都调好，最终相对当前坏版本应有明显提升；相对强 DINO-only baseline 则不要期待巨大跃迁，更现实目标是小到中等幅度提升。
