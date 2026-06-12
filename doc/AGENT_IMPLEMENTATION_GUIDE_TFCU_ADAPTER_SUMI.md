# Agent Implementation Guide: TFCU-Style Temporal Cue Unraveling + Task-Specific Adapter + SUMI-Style Losses

> Target task: video inpainting tamper localization / mask prediction.  
> Target baseline: current DINOv3 ViT-L/16 B23 + LoRA rank 32 + LiteBoundaryDecoder framework.  
> Scope: implement three independent experiment branches and one merged final branch without breaking previous CCM / FGM / HP3D / FPM experiments.

---

## 0. Why this change exists

The current framework already uses a strong foundation backbone and is good at cross-source / cross-dataset testing, but same-source DVI/CPNET performance is not consistently above existing methods. Current experiment notes show:

- `b23_ccm_lite_lora32`: stable baseline, best val IoU around `0.8218`, F1 around `0.8974`.
- `b23_ccm_fgm_lite_lora32_more`: best same-source val IoU around `0.8253`, but later epochs showed NaN risk.
- `b23_ccm_fgm_forensic_gated_lora32`: failed branch, best val IoU around `0.7719`; forensic branch alpha grew too large and over-interfered with the main features.

Therefore the new direction must **not** simply make FGM or forensic branches stronger. It should:

1. Replace raw FGM mask/cue bank with **multi-time-scale temporal cue unraveling** inspired by TFCU.
2. Add a **task-specific forensics adapter** that can be tested first on the clean baseline, with CCM and FGM disabled.
3. Add **sufficiency + minimality losses** inspired by SUMI-IFL and related top-conference work.
4. Provide a merged final version that combines all three safely.

---

## 1. Papers and ideas to borrow

### 1.1 TFCU: temporal cue unraveling

TFCU, *Face Forgery Video Detection via Temporal Forgery Cue Unraveling*, CVPR 2025, decomposes temporal forgery cues into three progressive levels:

- momentary anomaly,
- gradual inconsistency,
- cumulative distortion.

The original paper implements these through:

- consecutive correlate module,
- future guide module,
- historical review module.

For this project, do **not** copy TFCU as a face-video classifier. Adapt the idea to dense localization:

```text
Video feature sequence F[1:T]
  -> momentary anomaly map A_mom[1:T]
  -> gradual inconsistency map A_grad[1:T]
  -> cumulative distortion map A_cum[1:T]
  -> gated fusion into dense tamper feature/logit delta
```

Key difference from current FGM:

```text
Old FGM: store raw 16x16 historical forgery cue / mask bank.
New TCU: compute multi-scale temporal anomaly states inside the current video window; optional EMA state is feature-level and reset per video, never raw mask bank.
```

Reference:

```text
https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.html
https://github.com/zhenglab/TFCU
```

### 1.2 Forensics Adapter: task-specific adapter for foundation backbones

Forensics Adapter, CVPR 2025 extension, argues that treating CLIP only as a generic feature extractor is insufficient because forgery-related cues are entangled with unrelated knowledge. It introduces a lightweight adapter to learn task-specific forgery traces, especially blending boundaries, and interacts with foundation model tokens while preserving generalization.

For this project, adapt the principle to DINOv3 B23:

```text
DINO token / feature
+ lightweight artifact tokens from boundary / residual / noise cues
+ small task-specific interaction adapter
-> task-adapted feature for decoder
```

Important: this is **not** the old forensic branch. The old branch failed because its alpha grew too large and strongly perturbed the main representation. The new adapter must be low-alpha, gated, and independently testable with CCM/FGM disabled.

Reference:

```text
https://arxiv.org/abs/2411.19715
https://github.com/OUC-VAS/ForensicsAdapter
```

### 1.3 SUMI-IFL: sufficiency + minimality constraints

SUMI-IFL, AAAI 2025, introduces sufficiency-view and minimality-view constraints for image forgery localization. It encourages forged features to include comprehensive forgery clues while suppressing task-unrelated information.

For this project, adapt the loss to video inpainting localization:

- sufficiency: every useful view must be able to predict a coarse mask independently;
- minimality: the fused tamper feature should avoid source/scene/semantic overfitting;
- boundary and temporal losses should remain because dense localization needs accurate edges and stable videos.

Reference:

```text
https://ojs.aaai.org/index.php/AAAI/article/view/32054
https://arxiv.org/abs/2412.09981
```

### 1.4 RITA-style process awareness as optional context

RITA, CVPR 2026 Findings, reformulates manipulation localization as sequence prediction and argues that a one-shot binary mask collapses the structure of multi-step manipulation. We do not implement full RITA here, but the TCU branch should expose multiple intermediate temporal maps and optional auxiliary heads rather than only a single final mask.

Reference:

```text
https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhu_Revisiting_Image_Manipulation_Localization_under_Realistic_Manipulation_Scenarios_CVPRF_2026_paper.pdf
```

---

## 2. Non-breaking implementation rules

The agent must follow these rules.

### 2.1 Do not modify existing experiment behavior

Existing configs and runs must remain valid:

```text
runs/b23_ccm_lite_lora32
runs/b23_ccm_fgm_lite_lora32
runs/b23_ccm_fgm_lite_lora32_more
runs/b23_ccm_fgm_forensic_gated_lora32
configs already used by those runs
```

All new logic must be gated by new config keys with default `enabled: false`.

### 2.2 Add new files rather than rewriting old ones

Preferred new paths:

```text
models/modules/temporal_cue_unraveling.py
models/modules/task_forensics_adapter.py
losses/sumi_localization_losses.py
configs/experiments/b23_ccm_tfcu_unravel_lora32.yml
configs/experiments/b23_task_adapter_baseline_lora32.yml
configs/experiments/b23_sumi_losses_lora32.yml
configs/experiments/b23_tfcu_adapter_sumi_final_lora32.yml
tools/materialize_new_ablation_configs.py       # optional
```

Only touch existing model builders / loss builders minimally, e.g.:

```text
build_model(cfg)
  if cfg.model.task_forensics_adapter.enabled: attach adapter
  if cfg.model.temporal_cue_unraveling.enabled: attach TCU

build_loss(cfg)
  if cfg.loss.sumi.enabled: add SUMI-style terms
```

### 2.3 Backward compatibility defaults

If these keys are absent, old experiments must run exactly as before:

```yaml
temporal_cue_unraveling:
  enabled: false

task_forensics_adapter:
  enabled: false

loss:
  sumi:
    enabled: false
```

### 2.4 Shape conventions

Use the existing project conventions if they differ, but internally document all conversions.

Expected feature tensor formats:

```text
Backbone feature:      F = [B, T, C, Hf, Wf], usually Hf=Wf=32
Frame RGB input:       I = [B, T, 3, H, W], usually H=W=512
Mask logits output:    logits = [B, T or 1, 1, H, W] or current project equivalent
Low-res mask/logits:   logits32 = [B, T, 1, Hf, Wf]
```

If the current code only predicts the center/current frame, TCU should still compute features over all T frames and return the final/current frame feature by default.

---

## 3. Task 1: Replace FGM with TFCU-style Temporal Cue Unraveling

### 3.1 New module name

```python
class TemporalCueUnraveling(nn.Module):
    """
    Dense localization adaptation of TFCU.
    Replaces raw FGM mask/cue bank with multi-time-scale temporal cue maps.
    """
```

Suggested file:

```text
models/modules/temporal_cue_unraveling.py
```

### 3.2 Inputs and outputs

Input:

```python
features: Tensor  # [B, T, C, Hf, Wf]
lowres_logits: Optional[Tensor]  # [B, T, 1, Hf, Wf], detached when used for quality
video_ids: Optional[List[str]]
frame_indices: Optional[Tensor]
rgb: Optional[Tensor]  # [B, T, 3, H, W], optional for RGB residual cue
```

Output dictionary:

```python
{
  "feat_delta": Tensor,       # [B, T, C, Hf, Wf]
  "logit_delta32": Tensor,    # [B, T, 1, Hf, Wf]
  "momentary_map32": Tensor,  # [B, T, 1, Hf, Wf]
  "gradual_map32": Tensor,    # [B, T, 1, Hf, Wf]
  "cumulative_map32": Tensor, # [B, T, 1, Hf, Wf]
  "gate": Tensor,             # [B, T, 3, Hf, Wf] or [B,T,3,1,1]
  "quality": Tensor           # [B,T,1,1,1]
}
```

### 3.3 Branch A: Momentary Anomaly Module

Borrow TFCU's consecutive correlation idea, but make it dense.

Implementation options:

```text
F_t, F_{t-1}
  -> normalize along channel
  -> local dot / cosine similarity
  -> abs diff and channel-reduced correlation
  -> depthwise separable conv
  -> momentary anomaly feature A_mom
```

Pseudo-code:

```python
f0 = F[:, 1:]
f1 = F[:, :-1]
diff = torch.abs(f0 - f1)
cos = 1.0 - cosine_similarity(f0, f1, dim=2, keepdim=True)
x = torch.cat([diff_proj(diff), cos], dim=2)
A_mom[:, 1:] = mom_conv(x)
A_mom[:, 0] = A_mom[:, 1]
```

Do not use ground-truth masks. Do not use persistent mask bank.

### 3.4 Branch B: Gradual Inconsistency Module

Borrow TFCU's future guide idea. Instead of pushing raw masks into future frames, maintain an anomaly hidden state.

```text
h_t = ConvGRU(A_mom_t, h_{t-1})
quality_t = quality(lowres_logits_t, entropy_t)
h_t = quality_t * h_t + (1-quality_t) * detach(h_{t-1})
A_grad_t = grad_head([A_mom_t, h_t, F_t])
```

Important settings:

```yaml
gradual:
  hidden_dim: 128
  convgru_kernel: 3
  detach_history: true
  quality_gate: true
  min_quality: 0.25
```

Quality gate candidates:

```text
prediction confidence = mean(max(p, 1-p))
mask entropy = -p log p - (1-p) log(1-p)
area ratio validity = mask area within [0.002, 0.60]
temporal smoothness = IoU between adjacent low-res masks if available
```

### 3.5 Branch C: Cumulative Distortion Module

Borrow TFCU's historical review idea. Use bidirectional momentum accumulation over the current sampled video window.

Forward accumulation:

```python
s_fwd_t = momentum * s_fwd_{t-1} + (1-momentum) * A_grad_t
```

Backward review:

```python
s_bwd_t = momentum * s_bwd_{t+1} + (1-momentum) * A_grad_t
```

Cumulative cue:

```python
A_cum_t = cum_head(torch.cat([s_fwd_t, s_bwd_t, A_grad_t], dim=channel))
```

No raw cue memory. Optional feature EMA state may be used only if reset per video:

```yaml
stateful_eval:
  enabled: true
  state_type: feature_ema
  reset_on_new_video: true
  store_raw_mask: false
```

### 3.6 Fusion

Use a conservative residual delta.

```python
weights = softmax(gate_head([A_mom, A_grad, A_cum, F]), dim=branch)
A = w0*A_mom + w1*A_grad + w2*A_cum
feat_delta = alpha * proj(A)
F_out = F + feat_delta
logit_delta32 = logit_head(A)
```

Hard constraints:

```yaml
alpha_init: 0.001
alpha_max: 0.035
clamp_alpha: true
branch_dropout: 0.10
```

### 3.7 TCU-specific auxiliary losses

Each branch gets weak mask supervision at 32x32:

```yaml
loss:
  tcu_momentary_mask32: 0.015
  tcu_gradual_mask32: 0.020
  tcu_cumulative_mask32: 0.020
  tcu_branch_diversity: 0.005
  temporal_consistency: 0.030
```

`branch_diversity` can be decorrelation between branch logits/features to avoid all branches becoming identical.

### 3.8 How this replaces old FGM

For TCU experiments:

```yaml
tfcu:
  fgm:
    enabled: false

fgm_bank:
  stateful_train: false
  stateful_eval: false
  store_raw_cue: false

temporal_cue_unraveling:
  enabled: true
```

Do not remove old FGM code. Keep old configs intact.

---

## 4. Task 2: Task-Specific Forensics Adapter on clean baseline first

### 4.1 Experimental requirement

First run this adapter **without CCM and without FGM**. This isolates whether the large backbone lacks task-specific adaptation.

The first adapter config must satisfy:

```yaml
tfcu:
  ccm:
    enabled: false
  fgm:
    enabled: false

temporal_cue_unraveling:
  enabled: false

task_forensics_adapter:
  enabled: true
```

### 4.2 New module name

```python
class TaskSpecificForensicsAdapter(nn.Module):
    """
    Low-alpha side-car adapter for DINO features.
    Learns inpainting localization cues: boundary, residual, local artifact, temporal residual token.
    """
```

Suggested file:

```text
models/modules/task_forensics_adapter.py
```

### 4.3 Adapter design

Use a side-car adapter after backbone output first, because it is least invasive and easiest to debug.

```text
DINO feature F        [B,T,C,32,32]
RGB frames           [B,T,3,512,512]
  -> fixed residual stem: SRM / gradient / Laplacian / optional frame difference
  -> residual tokens R  [B,T,Cr,32,32]
  -> adapter cross-attention: Query=F, Key/Value=R + learnable forgery tokens
  -> gated residual delta
  -> F_adapted = F + alpha * gate * adapter_delta
```

Recommended minimal implementation:

```python
class TaskSpecificForensicsAdapter(nn.Module):
    def __init__(self, in_dim, adapter_dim=64, out_dim=None, num_prompt_tokens=4):
        self.residual_stem = FixedResidualStem(...)       # fixed SRM/Laplacian/gradient
        self.res_proj = nn.Conv2d(res_ch, adapter_dim, 1)
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompt_tokens, adapter_dim) * 0.02)
        self.q_proj = nn.Conv2d(in_dim, adapter_dim, 1)
        self.k_proj = nn.Conv2d(adapter_dim, adapter_dim, 1)
        self.v_proj = nn.Conv2d(adapter_dim, adapter_dim, 1)
        self.out_proj = nn.Conv2d(adapter_dim, in_dim, 1)
        self.gate = nn.Sequential(nn.Conv2d(in_dim + adapter_dim, 1, 1), nn.Sigmoid())
        self.alpha = LearnableClampedScalar(init=0.001, max=0.040)
```

### 4.4 Optional backbone-internal adapter

If the repo can expose intermediate DINO layers, add this later:

```yaml
task_forensics_adapter:
  insertion_mode: selected_layers
  insert_layers: [6, 12, 18, 23]
```

But the first implementation should support:

```yaml
insertion_mode: post_backbone
```

This avoids breaking DINO internals.

### 4.5 Adapter objectives

The adapter should expose weak auxiliary outputs:

```python
adapter_outputs = {
  "adapter_mask32": ...,       # [B,T,1,32,32]
  "adapter_boundary32": ...,   # [B,T,1,32,32]
  "adapter_gate": ...,
  "adapter_alpha": ...
}
```

Loss weights:

```yaml
adapter_mask32: 0.020
adapter_boundary32: 0.030
adapter_gate_l1: 0.001
```

### 4.6 Safety constraints

The old forensic-gated branch failed because low-level branch alpha grew too large. Enforce:

```yaml
task_forensics_adapter:
  alpha_init: 0.001
  alpha_max: 0.040
  lr: 1.0e-5
  drop_path: 0.15
  gate_bias_init: -2.0
  warmup_epochs: 10
```

Training schedule:

```text
Epoch 0-10: adapter alpha grows slowly; no SUMI minimality yet.
Epoch 10-30: adapter + decoder train.
After epoch 30: LoRA can train with low LR if stable.
```

---

## 5. Task 3: SUMI-style sufficiency + minimality losses

### 5.1 New loss file

```text
losses/sumi_localization_losses.py
```

### 5.2 Keep existing base losses

Do not remove existing loss:

```yaml
base:
  dice: 1.00
  bce: 0.50
  tversky: 0.20
  boundary: 0.20
```

### 5.3 Sufficiency-view losses

Each view should be individually predictive. Use weak auxiliary supervision.

Candidate views:

```text
main_logits
adapter_mask32
adapter_boundary32
tcu_momentary_map32
tcu_gradual_map32
tcu_cumulative_map32
```

Implementation:

```python
def sufficiency_loss(aux_logits: Dict[str, Tensor], gt_mask):
    loss = 0
    for name, logit in aux_logits.items():
        gt32 = resize_mask(gt_mask, logit.shape[-2:])
        loss += weight[name] * (dice_loss(logit, gt32) + 0.5 * bce(logit, gt32))
    return loss
```

Recommended weights:

```yaml
sufficiency:
  adapter_mask32: 0.020
  adapter_boundary32: 0.030
  tcu_momentary_mask32: 0.015
  tcu_gradual_mask32: 0.020
  tcu_cumulative_mask32: 0.020
```

### 5.4 Minimality-view losses

Minimality should suppress task-irrelevant information without destroying useful dataset-specific cues. Use weak losses and delayed schedule.

Recommended components:

#### A. Variational information bottleneck on fused tamper feature

```python
mu, logvar = bottleneck_head(fused_feature)
z = mu + eps * exp(0.5 * logvar)
kl = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
```

Config:

```yaml
minimality:
  ib_kl_weight: 0.0002
  start_epoch: 20
```

#### B. Source adversarial loss with Gradient Reversal

This discourages the fused tamper representation from becoming a pure DVI/CPNET source classifier.

```python
source_pred = source_classifier(GRL(global_pool(fused_tamper_feature)))
loss_source_adv = CE(source_pred, source_id)
```

Config:

```yaml
source_adversarial:
  enabled: true
  weight: 0.010
  grl_lambda: 0.05
  start_epoch: 30
```

Keep this weak. Strong source-invariance may hurt same-source results. The point is to reduce semantic/source shortcuts, not erase all useful artifacts.

#### C. Background compactness / authentic-region suppression

Use GT authentic pixels to suppress false positive activation.

```python
bg = 1 - gt_mask32
loss_bg = mean(abs(tamper_feature_activation) * bg)
```

Config:

```yaml
background_suppression:
  enabled: true
  weight: 0.010
```

### 5.5 Boundary and temporal losses

Keep and strengthen boundary supervision modestly because same-source failure often comes from edge / small-region errors.

```yaml
boundary:
  weight: 0.25
  band_width: 5
  focal_weight: 0.05

small_region_focal:
  enabled: true
  weight: 0.05
  area_threshold: 0.02
```

Temporal consistency:

```yaml
temporal_consistency:
  enabled: true
  weight: 0.03
  mode: adjacent_lowres
  detach_target: true
```

If optical flow is available, use warp consistency. If not, use adjacent low-res mask consistency and feature consistency.

### 5.6 Loss warmup schedule

Avoid destabilizing the already sensitive training.

```yaml
loss_schedule:
  epoch_0_10:
    base_only: true
  epoch_10_20:
    enable_sufficiency: true
  epoch_20_30:
    enable_information_bottleneck: true
  epoch_30_plus:
    enable_source_adversarial: true
```

---

## 6. Final merged architecture

The final version combines all three directions:

```text
RGB video frames
  -> DINOv3 B23 + LoRA
  -> TaskSpecificForensicsAdapter
  -> optional CCM-Lite
  -> TemporalCueUnraveling, replacing raw FGM bank
  -> LiteBoundaryDecoder
  -> base + SUMI-style losses
```

Recommended final rules:

```yaml
tfcu:
  ccm:
    enabled: true
    alpha_init: 0.002
    alpha_max: 0.050
  fgm:
    enabled: false

fgm_bank:
  stateful_train: false
  stateful_eval: false
  store_raw_cue: false

task_forensics_adapter:
  enabled: true
  alpha_max: 0.035

temporal_cue_unraveling:
  enabled: true
  alpha_max: 0.030

loss:
  sumi:
    enabled: true
```

If this final version hurts OPN or becomes unstable, disable in this order:

```text
1. source_adversarial minimality
2. cumulative branch stateful eval
3. adapter alpha_max from 0.035 to 0.020
4. CCM alpha_max from 0.050 to 0.030
```

---

## 7. Required YAML configs

The delivered YAML files must be copied into:

```text
configs/experiments/
```

Required configs:

```text
b23_ccm_tfcu_unravel_lora32.yml
b23_task_adapter_baseline_lora32.yml
b23_sumi_losses_lora32.yml
b23_tfcu_adapter_sumi_final_lora32.yml
```

Each config must run independently and must not import/overwrite previous config files unless the project already has a standard include mechanism.

---

## 8. Suggested experiment order

Run in this order:

```text
E1: b23_task_adapter_baseline_lora32.yml
    Goal: prove task adapter helps baseline without CCM/FGM.

E2: b23_sumi_losses_lora32.yml
    Goal: isolate loss changes on a conservative CCM-only model.

E3: b23_ccm_tfcu_unravel_lora32.yml
    Goal: replace FGM bank with temporal cue unraveling.

E4: b23_tfcu_adapter_sumi_final_lora32.yml
    Goal: combine adapter + TCU + SUMI losses.
```

Do not judge only by average IoU. Report:

```text
DVI_20 IoU / F1 / Precision / Recall
CPNET_20 IoU / F1 / Precision / Recall
OPN_20 IoU / F1 / Precision / Recall
same_source_avg = (DVI_20 + CPNET_20) / 2
cross_source = OPN_20
pareto_score = same_source_avg + 0.7 * cross_source
adapter_alpha
tcu_alpha
momentary/gradual/cumulative gate means
source_adv_loss
ib_kl
```

Expected success criteria:

```text
E1 adapter baseline > old CCM-only or at least improves same-source boundary F1.
E3 TCU > old FGM bank on same-source, with no NaN.
E4 final same_source_avg improves while OPN is not below CCM-only by more than 0.5 IoU.
```

---

## 9. Agent checklist

### 9.1 Locate current code

Use these commands in the repo:

```bash
rg "class .*FGM|fgm|fgm_bank|cue_feedback" -n .
rg "class .*CCM|ccm|alpha_cc" -n .
rg "LiteBoundaryDecoder|BoundaryDecoder|decoder" -n models .
rg "dice|tversky|boundary|build_loss|loss" -n .
rg "DINO|dinov3|lora|rank" -n .
rg "config.resolved|yaml|yml|OmegaConf|argparse" -n .
```

### 9.2 Add modules

Add:

```text
models/modules/temporal_cue_unraveling.py
models/modules/task_forensics_adapter.py
losses/sumi_localization_losses.py
```

### 9.3 Register modules conditionally

Do this only inside config-gated branches.

Pseudo-code:

```python
if cfg.get("task_forensics_adapter", {}).get("enabled", False):
    self.task_adapter = TaskSpecificForensicsAdapter(...)
else:
    self.task_adapter = None

if cfg.get("temporal_cue_unraveling", {}).get("enabled", False):
    self.tcu = TemporalCueUnraveling(...)
else:
    self.tcu = None
```

Forward:

```python
features = backbone(frames)
aux = {}

if self.task_adapter is not None:
    features, adapter_aux = self.task_adapter(features, frames)
    aux.update(adapter_aux)

if self.ccm is not None and cfg.tfcu.ccm.enabled:
    features, ccm_aux = self.ccm(features, ...)
    aux.update(ccm_aux)

if self.tcu is not None:
    features, tcu_aux = self.tcu(features, lowres_logits=None, rgb=frames, video_ids=video_ids)
    aux.update(tcu_aux)

# old fgm only runs when cfg.tfcu.fgm.enabled == true
if self.fgm is not None and cfg.tfcu.fgm.enabled:
    features, fgm_aux = self.fgm(features, ...)
    aux.update(fgm_aux)

logits = decoder(features, aux=aux)
return {"logits": logits, **aux}
```

### 9.4 Unit tests / sanity tests

Minimum tests:

```bash
python - <<'PY'
import torch
from models.modules.temporal_cue_unraveling import TemporalCueUnraveling
m = TemporalCueUnraveling(in_dim=256, hidden_dim=128)
x = torch.randn(2, 4, 256, 32, 32)
y = m(x)
print({k: tuple(v.shape) if hasattr(v, 'shape') else type(v) for k,v in y.items()})
PY

python - <<'PY'
import torch
from models.modules.task_forensics_adapter import TaskSpecificForensicsAdapter
m = TaskSpecificForensicsAdapter(in_dim=256, adapter_dim=64)
f = torch.randn(2, 4, 256, 32, 32)
rgb = torch.randn(2, 4, 3, 512, 512)
y, aux = m(f, rgb)
print(y.shape, aux.keys())
PY
```

Full training smoke test:

```bash
python train.py --config configs/experiments/b23_task_adapter_baseline_lora32.yml --max_epochs 2
python train.py --config configs/experiments/b23_sumi_losses_lora32.yml --max_epochs 2
python train.py --config configs/experiments/b23_ccm_tfcu_unravel_lora32.yml --max_epochs 2
python train.py --config configs/experiments/b23_tfcu_adapter_sumi_final_lora32.yml --max_epochs 2
```

### 9.5 Required logging

Add these metrics to train/val logs:

```text
adapter_alpha
adapter_gate_mean
adapter_mask32_loss
adapter_boundary32_loss
tcu_alpha
tcu_gate_momentary_mean
tcu_gate_gradual_mean
tcu_gate_cumulative_mean
tcu_momentary_loss
tcu_gradual_loss
tcu_cumulative_loss
sumi_sufficiency_loss
sumi_ib_kl
sumi_source_adv_loss
background_suppression_loss
```

---

## 10. Failure diagnosis

### 10.1 Adapter branch improves train but hurts OPN

Try:

```yaml
task_forensics_adapter:
  alpha_max: 0.020
  drop_path: 0.25
loss:
  sumi:
    source_adversarial:
      enabled: true
      weight: 0.015
```

### 10.2 TCU has no gain over FGM

Try:

```yaml
temporal_cue_unraveling:
  total_frames: 8
  momentary:
    use_rgb_residual: true
  gradual:
    hidden_dim: 192
  cumulative:
    momentum: 0.95
loss:
  sumi:
    sufficiency:
      tcu_gradual_mask32: 0.030
      tcu_cumulative_mask32: 0.030
```

### 10.3 Training unstable / NaN

Try:

```yaml
optimizer:
  grad_clip_norm: 0.5
nan_guard:
  enabled: true
  clamp_logits: [-20.0, 20.0]
temporal_cue_unraveling:
  alpha_max: 0.020
task_forensics_adapter:
  alpha_max: 0.020
loss:
  sumi:
    minimality:
      ib_kl_weight: 0.0001
    source_adversarial:
      enabled: false
```

### 10.4 Same-source improves but cross-source drops

Try:

```yaml
loss:
  sumi:
    source_adversarial:
      enabled: true
      weight: 0.015
      grl_lambda: 0.08
task_forensics_adapter:
  gate_bias_init: -2.5
temporal_cue_unraveling:
  fusion:
    gate_bias_init: -2.0
```

---

## 11. Deliverables expected from the agent

After implementation, the agent must provide:

```text
1. New/modified file list.
2. Four new configs copied into configs/experiments/.
3. Smoke test output for all four configs.
4. Confirmation that old configs still parse and run for one forward pass.
5. A short ablation table template with metrics listed in section 8.
```

