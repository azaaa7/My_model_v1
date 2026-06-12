# Agent Implementation Guide: RGFGM + HP3D/Noise Adapter + Prototype Memory

> Agent start here. Search keyword: `RGFGM_HP3D_FPM_V1`.
>
> Target config file: `configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml`.
>
> Target run name: `b23_ccm_rgfgm_hp3d_fpm_lora32`.

This guide tells a coding agent how to modify the existing video inpainting tamper localization framework so the new config can run end-to-end. The implementation must preserve the current strong CCM/FGM behavior while adding three modules:

1. **CCM 主干 + Reliability-Gated FGM**
2. **HP3D/Noise Adapter with strongly constrained alpha**
3. **Prototype Memory replacing raw FGM cue bank**

The new modules must be switchable from YAML so the ablation presets in `b23_ccm_rgfgm_hp3d_fpm_lora32.yml` can be materialized into separate experiments.

---

## 0. Why this design

Current experiment evidence:

- `b23_ccm_lite_lora32` is the most stable strong baseline; it reached `val_iou=0.8218`, `val_f1=0.8974`.
- `b23_ccm_fgm_lite_lora32` shows FGM is useful but modest: `val_iou=0.8234`, `val_f1=0.8925`.
- `b23_ccm_fgm_lite_lora32_more` reaches the best current same-source validation IoU, `val_iou=0.8253`, but later has NaN risk.
- `b23_ccm_fgm_forensic_gated_lora32` fails badly: `val_iou=0.7719`, and its `forensic_branch.alpha` grows to `0.0933`, indicating low-level forensic features over-intervene.

Therefore the new version should start from the stable CCM+FGM design, keep `bank_len=5/topk=128/diff_scale=1.0`, but add conservative gates and small-alpha auxiliary evidence instead of directly strengthening FGM or forensic features.

Paper/code practices to borrow:

- **TruFor**: RGB + learned noise-sensitive fingerprint, transformer fusion, and an explicit reliability map for error-prone localization regions. Borrow the reliability-gating idea, not the full architecture.
- **MVSS-Net**: semantic-agnostic boundary artifacts + noise view with multi-scale pixel/edge/image supervision. Borrow boundary/noise supervision and conservative noise features.
- **TruVIL**: multi-scale 3D high-pass noise extraction, cross-modality attentive fusion, attentive noise decoding for video inpainting localization. Borrow HP3D/noise extraction and gated cross-modal fusion.
- **XMem/Cutie**: memory should be consolidated and object/prototype-level, because raw dense pixel memory can be noisy and distractor-sensitive. Borrow prototype-level memory read/write instead of raw cue-map propagation.

---

## 1. First locate the existing code

Run these search commands at the repository root:

```bash
rg -n "class .*TFCU|TFCU|ccm|alpha_cc|fgm|cue_feedback|fgm_bank|quality_gate|forensic_branch|LiteBoundaryDecoder" .
rg -n "config\.resolved|OmegaConf|yaml|argparse|train_full_video_windows|stateful_eval|reset_on_new_video" .
rg -n "best_iou|val_iou|val_f1|precision|recall|boundary|tversky" .
```

Likely places to edit, but confirm by grep rather than assuming exact names:

```text
configs/experiments/                         # add new yml here
models/ or model/                            # TFCU / CCM / FGM / decoder modules
models/modules/ or networks/modules/          # add reliability gate, noise adapter, prototype memory
train.py / engine.py / trainer.py             # optimizer param groups, logging, NaN guard
evaluate.py / test.py                         # DVI/CPNET/OPN reporting and stateful reset
utils/config.py / config.py                   # allow new config keys or runtime-only extraction
```

The code must keep all old configs runnable. Do not delete the old FGM raw cue bank path.

---

## 2. Config-loader requirement

The new YAML contains two top-level sections:

```yaml
meta: ...
runtime: ...
ablation_presets: ...
```

If the current training script expects a flat runtime config, modify the loader to do one of the following:

```python
cfg_all = load_yaml(path)
cfg = cfg_all.get("runtime", cfg_all)
```

`meta` and `ablation_presets` should not be passed into the model unless the code explicitly uses them for experiment generation.

Optional helper script:

```bash
python tools/materialize_ablation_configs.py \
  --config configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml \
  --out_dir configs/experiments/rgfgm_hp3d_fpm_ablation/
```

This script should deep-copy `runtime`, apply each `ablation_presets.*.overrides`, and write one normal runtime-only YAML per ablation.

---

## 3. Target architecture

The forward path should be:

```text
RGB clip/video windows
  -> DINOv3 ViT-L B23 + LoRA
  -> CCM-Lite
  -> base low-res feature F_ccm
  -> FGM candidate delta ΔF_fgm / Δlogit_fgm
  -> ReliabilityGate produces g_fgm
  -> PrototypeMemory read/write produces ΔF_proto
  -> HP3D/SRM/Bayar NoiseAdapter produces ΔF_noise and reliability_noise
  -> gated fusion: F = F_ccm + g_fgm * ΔF_fgm + g_proto * ΔF_proto + alpha_noise * g_noise * ΔF_noise
  -> LiteBoundaryDecoder + boundary refiner
  -> logits/masks + aux outputs + logging scalars
```

Important constraints:

```text
Reliability gate must be initialized conservative.
Noise adapter alpha must be capped at 0.02 by default.
Prototype memory must not write low-confidence masks.
All three new modules must be detachable/disable-able by config.
NaN guard and alpha/feedback clamps are mandatory.
```

---

## 4. Module 1: Reliability-Gated FGM

### 4.1 Expected behavior

Current FGM is useful but small. The new gate should let FGM help when temporal cues are reliable and automatically reduce FGM influence when cues look noisy or out-of-domain.

Formula:

```text
F_out = F_ccm + g_fgm * ΔF_fgm
or
logits_out = logits_ccm + g_fgm * Δlogits_fgm
```

Prefer feature-level fusion if the existing FGM is feature-level. Use logit-level fusion only if current FGM already outputs an auxiliary mask/logit.

### 4.2 Reliability inputs

Implement a `ReliabilityGate` class that can consume low-res maps/scalars. Required inputs:

```text
mask_entropy: uncertainty of current predicted probability map
prev_iou: IoU between current mask and previous or warped previous mask
data/noise confidence: from NoiseAdapter if enabled, otherwise constant 0.5
motion residual: optional; if no flow/residual exists, use zero tensor
```

Low-res entropy:

```python
def binary_entropy(prob, eps=1e-6):
    prob = prob.clamp(eps, 1 - eps)
    ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    return ent / 0.69314718056
```

Gate initialization:

```python
# bias_init=-2.0 gives sigmoid ~0.12 before gate_max clamp.
g = sigmoid(conv_or_mlp(x) + bias_init) * gate_max
```

Config keys to support:

```yaml
reliability_gate:
  enabled: true
  gate_bias_init: -2.0
  gate_max: 0.70
  inputs: ...
```

### 4.3 Implementation details

Add something like:

```python
class ReliabilityGate(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, gate_max=0.7, bias_init=-2.0):
        super().__init__()
        self.gate_max = gate_max
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, bias_init)

    def forward(self, gate_inputs):
        x = torch.cat(gate_inputs, dim=1)
        return torch.sigmoid(self.net(x)) * self.gate_max
```

When `detach_inputs=true`, detach only the gate evidence tensors, not the final FGM delta.

Log at least:

```text
reliability_gate.mean
reliability_gate.mean_DVI
reliability_gate.mean_CPNET
reliability_gate.mean_OPN
```

Expected successful pattern:

```text
gate mean on OPN < gate mean on DVI/CPNET
same_source_avg increases
OPN does not drop below the stable CCM/FGM reference
```

---

## 5. Module 2: HP3D/Noise Adapter with constrained alpha

### 5.1 Do not reuse the failed forensic branch directly

The previous forensic-gated run failed because the forensic branch became too strong. The new adapter must be a small evidence branch:

```text
alpha_init: 0.001
alpha_max: 0.020
lr: 1e-5
detach_stem_epochs: 50
drop_path: 0.20
fusion: gated_cross_attention
```

Alpha must be parameterized with a hard cap:

```python
class CappedAlpha(nn.Module):
    def __init__(self, init=0.001, max_value=0.02):
        super().__init__()
        init_ratio = min(max(init / max_value, 1e-4), 1 - 1e-4)
        self.raw = nn.Parameter(torch.tensor(math.log(init_ratio / (1 - init_ratio))))
        self.max_value = max_value
    def forward(self):
        return torch.sigmoid(self.raw) * self.max_value
```

### 5.2 Noise extraction

Implement a new `HP3DNoiseAdapter` or similar module. It should include:

```text
SRM filters: fixed 2D high-pass filters per frame
BayarConv: constrained high-pass convolution, optional trainable
HP3D: fixed temporal-spatial high-pass kernels over [B, T, C, H, W]
Tiny projection: Conv3d/Conv2d -> GroupNorm -> GELU -> 1x1 projection to 32 channels
```

Suggested input/output shape:

```text
input frames: [B, T, 3, H, W]
noise feature high-res: [B, T, 32, H/4, W/4] or [B*T, 32, H/4, W/4]
project to low-res: [B*T, C_dino, 32, 32] or adapter delta compatible with decoder feature
reliability map: [B*T, 1, 32, 32]
```

### 5.3 Fusion

Use gated cross-attention or a safe approximation:

```python
noise_delta = self.noise_proj(noise_feat)
g_noise = self.noise_gate(torch.cat([main_feat, noise_delta], dim=1))
F = F + alpha_noise * g_noise * noise_delta
```

If implementing full cross-attention is risky, first implement the safe additive gated projection above. Keep the class name/config field as `gated_cross_attention` but allow a `lite` code path.

Log:

```text
forensic_adapter.alpha
forensic_adapter.gate_mean
forensic_adapter.reliability_mean
```

Fail-safe checks:

```text
alpha must never exceed alpha_max.
If alpha > 0.03, implementation is wrong for the default config.
If OPN drops sharply, run A7 with alpha_max=0.01.
```

---

## 6. Module 3: Prototype Memory replacing raw FGM cue bank

### 6.1 Motivation

Do not store full dense 16x16 cue maps as the only memory. Store compact foreground/background prototypes so the model reads summarized evidence instead of replaying source-specific mask patterns.

### 6.2 Data structure

For each video in stateful train/eval:

```python
memory = {
  "fg_proto": Tensor[num_fg_proto, C],
  "bg_proto": Tensor[num_bg_proto, C],
  "fg_valid": Tensor[num_fg_proto],
  "bg_valid": Tensor[num_bg_proto],
  "last_video_id": str,
}
```

Reset on new video:

```python
if cfg.fgm_bank.reset_on_new_video and video_id != last_video_id:
    memory.reset()
```

### 6.3 Prototype update

From low-res feature `F` and mask probability `P`:

```python
P_fg = P.detach()
P_bg = 1 - P_fg
p_fg = weighted_avg(F.detach(), P_fg)
p_bg = weighted_avg(F.detach(), P_bg)
```

Write only if all conditions are satisfied:

```text
mean confidence >= 0.75
mean entropy <= 0.35
area ratio in [0.002, 0.600]
fgm global gate >= 0.15
current epoch >= 10
```

Momentum update:

```python
proto = momentum * proto + (1 - momentum) * new_proto
proto = F.normalize(proto, dim=-1)
```

### 6.4 Prototype readout

Use foreground-background masked attention:

```python
q = proj_q(F)                         # [B, HW, C]
k = proj_k(torch.cat([fg_proto, bg_proto]))
v = proj_v(torch.cat([fg_proto, bg_proto]))
attn = softmax(q @ k.T / temperature)
proto_ctx = attn @ v
proto_ctx = proto_ctx.reshape(B, C, H, W)
ΔF_proto = out_proj(proto_ctx)
F = F + g_proto * ΔF_proto
```

Use `read_gate_bias_init=-1.5`, `read_gate_max=0.60`.

OOD policy:

```text
If reliability_gate global mean < 0.15, memory is read-only and must not write.
```

Log:

```text
prototype_memory.write_rate
prototype_memory.read_gate_mean
prototype_memory.fg_proto_norm
prototype_memory.bg_proto_norm
```

---

## 7. Losses and optimizer

Use the new default weights:

```yaml
loss:
  dice: 1.00
  bce: 0.50
  tversky: 0.20
  boundary: 0.20
  boundary_focal: 0.05
  temporal_consistency: 0.03
  fgm_mask32: 0.015
  noise_mask32: 0.015
  noise_boundary: 0.020
  gate_regularization: 0.002
```

Parameter-group learning rates:

```yaml
backbone_lora: 1.0e-4
decoder: 1.0e-4
ccm: 2.0e-5
fgm: 5.0e-5
reliability_gate: 2.0e-5
prototype_memory: 1.0e-5
forensic_adapter: 1.0e-5
```

Add NaN guard:

```text
clip grad norm to 1.0
clamp logits to [-30, 30]
skip batch if loss is NaN/Inf
stop run after 3 consecutive NaN batches
never overwrite best_iou.pt with NaN validation
```

---

## 8. Ablation configs to materialize

The YAML includes these ablation presets:

| ID | Purpose | Expected result |
|---|---|---|
| A0 | Existing CCM-only reference | Stable F1 reference |
| A1 | Existing standard FGM reference | Stable IoU reference |
| A2 | Reliability-Gated FGM only | Same-source up, OPN protected |
| A3 | RGFGM + HP3D/Noise | Boundary/noise improvement, alpha safe |
| A4 | RGFGM + Prototype Memory | Tests raw bank replacement |
| A5 | Full model | Main proposed experiment |
| A6 | Full without reliability gate | Failure-control; likely hurts OPN |
| A7 | Full with alpha_max=0.01 | Conservative OPN-safe variant |
| A8 | Full with alpha_max=0.03 | Aggressive noise variant; run after stability confirmed |
| A9 | Gate bias -1.0 | More same-source gain, higher OPN risk |
| A10 | Gate bias -3.0 | OPN-safe conservative gate |

For a strict parser, generate runtime-only files:

```bash
configs/experiments/rgfgm_hp3d_fpm_ablation/A2_reliability_gated_fgm.yml
configs/experiments/rgfgm_hp3d_fpm_ablation/A3_rgfgm_plus_hp3d_noise.yml
configs/experiments/rgfgm_hp3d_fpm_ablation/A4_rgfgm_plus_prototype_memory.yml
configs/experiments/rgfgm_hp3d_fpm_ablation/A5_full_rgfgm_hp3d_fpm.yml
...
```

---

## 9. Minimal unit tests before training

Create or run a smoke test using random tensors:

```python
B, T, C, H, W = 1, 4, 3, 512, 512
frames = torch.randn(B, T, C, H, W).cuda()
mask = torch.rand(B, T, 1, H, W).cuda()
batch = {"frames": frames, "mask": mask, "video_id": ["dummy"], "frame_idx": torch.arange(T)}
out = model(batch)
assert torch.isfinite(out["logits"]).all()
assert out["logits"].shape[-2:] == (512, 512)
```

Test each ablation switch:

```text
FGM disabled: model still runs.
Reliability gate disabled: old FGM path still runs.
Noise adapter disabled: no missing key in gate inputs.
Prototype memory disabled: raw cue compatibility path still runs.
State reset: two video_ids must not share memory.
```

---

## 10. Training and evaluation commands

Main run:

```bash
python train.py --config configs/experiments/b23_ccm_rgfgm_hp3d_fpm_lora32.yml
```

Materialized ablation run example:

```bash
python train.py --config configs/experiments/rgfgm_hp3d_fpm_ablation/A5_full_rgfgm_hp3d_fpm.yml
```

Test suite should report:

```text
DVI_20 IoU/F1/Precision/Recall/Loss
CPNET_20 IoU/F1/Precision/Recall/Loss
OPN_20 IoU/F1/Precision/Recall/Loss
Average
same_source_avg = (DVI_20 + CPNET_20) / 2
cross_source_opn = OPN_20
pareto_score = same_source_avg + 0.7 * cross_source_opn
```

Do not accept a same-source gain that destroys OPN. Preferred selection rule:

```text
same_source_avg >= current stable FGM baseline
OPN_20 >= CCM-only or stable FGM OPN reference
pareto_score improves
reliability_gate.mean_OPN < reliability_gate.mean_DVI/CPNET
forensic_adapter.alpha <= 0.02 by default
```

---

## 11. Definition of done

The implementation is complete when:

1. Existing configs still run without code changes.
2. The new YAML loads from `runtime` and starts training.
3. A2/A3/A4/A5 ablations can be materialized and run.
4. Logs include all new gate/alpha/prototype metrics.
5. No NaN validation checkpoint overwrites `best_iou.pt`.
6. `forensic_adapter.alpha` is capped by `alpha_max`.
7. Prototype memory resets on video changes and does not write low-confidence masks.
8. OPN evaluation is included in the test suite for every proposed run.

