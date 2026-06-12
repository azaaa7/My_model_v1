from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_entropy(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    prob = prob.float().clamp(eps, 1.0 - eps)
    ent = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
    return ent / 0.69314718056


class CappedAlpha(nn.Module):
    def __init__(self, init: float = 0.001, max_value: float = 0.02):
        super().__init__()
        max_value = max(float(max_value), 1.0e-8)
        init_ratio = min(max(float(init) / max_value, 1.0e-4), 1.0 - 1.0e-4)
        raw = math.log(init_ratio / (1.0 - init_ratio))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.max_value = float(max_value)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) * self.max_value


class ReliabilityGate(nn.Module):
    """Conservative spatial gate for FGM features."""

    def __init__(self, cfg: dict[str, Any] | None = None, in_channels: int = 4):
        super().__init__()
        cfg = cfg or {}
        hidden_dim = int(cfg.get("hidden_dim", 32))
        self.enabled = bool(cfg.get("enabled", True))
        self.gate_max = float(cfg.get("gate_max", 0.70))
        self.detach_inputs = bool(cfg.get("detach_inputs", True))
        bias_init = float(cfg.get("gate_bias_init", -2.0))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, bias_init)

    def forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        if not inputs:
            raise ValueError("ReliabilityGate requires at least one input tensor")
        if self.detach_inputs:
            inputs = [x.detach() for x in inputs]
        x = torch.cat(inputs, dim=1)
        gate = torch.sigmoid(self.net(x)) * self.gate_max
        debug = {
            "reliability_gate_mean": float(gate.detach().float().mean().cpu()),
            "reliability_gate_min": float(gate.detach().float().amin().cpu()),
            "reliability_gate_max": float(gate.detach().float().amax().cpu()),
        }
        return gate, debug


class HP3DNoiseAdapter(nn.Module):
    """Small high-pass/noise evidence adapter with a hard-capped alpha."""

    def __init__(self, cfg: dict[str, Any] | None = None, out_channels: int = 128, target_resolution: int = 32):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.detach_input = bool(cfg.get("detach_input", True))
        self.target_resolution = int(cfg.get("target_resolution", target_resolution))
        hidden = int(cfg.get("hidden_channels", 32))
        self.alpha = CappedAlpha(float(cfg.get("alpha_init", 0.001)), float(cfg.get("alpha_max", 0.02)))
        self.alpha_max = float(cfg.get("alpha_max", 0.02))
        self.drop_path = float(cfg.get("drop_path", 0.20))

        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
                [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        kernels = kernels[:, None] / kernels.abs().flatten(1).sum(dim=1).clamp_min(1.0)[:, None, None, None]
        self.register_buffer("hp_kernels", kernels, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden)), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=max(1, hidden), bias=False),
            nn.GroupNorm(max(1, min(8, hidden)), hidden),
            nn.GELU(),
        )
        self.delta_proj = nn.Conv2d(hidden, int(out_channels), 1)
        self.gate = nn.Sequential(
            nn.Conv2d(int(out_channels) + hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.mask_head = nn.Conv2d(hidden, 1, 1)
        self.boundary_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def _extract_high_pass(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, K, 3, H, W]
        b, k, c, h, w = frames.shape
        x = frames.detach() if self.detach_input else frames
        gray = x.float().mean(dim=2).reshape(b * k, 1, h, w)
        hp = F.conv2d(gray, self.hp_kernels.to(dtype=gray.dtype), padding=1)
        temporal = torch.zeros_like(gray)
        if k > 1:
            seq = gray.reshape(b, k, 1, h, w)
            temporal[:, :, :, :] = (seq - seq.mean(dim=1, keepdim=True)).reshape(b * k, 1, h, w)
        return torch.cat([hp[:, :3], temporal], dim=1)

    def extract_features(self, frames: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, k = frames.shape[:2]
        hp = self._extract_high_pass(frames)
        hp = F.adaptive_avg_pool2d(hp, (self.target_resolution, self.target_resolution))
        noise_feat = self.stem(hp.to(dtype=dtype))
        mask32 = self.mask_head(noise_feat).reshape(b, k, 1, self.target_resolution, self.target_resolution)
        boundary32 = self.boundary_head(noise_feat).reshape(b, k, 1, self.target_resolution, self.target_resolution)
        noise_feat = noise_feat + (mask32.sum() + boundary32.sum()) * 0.0
        reliability = (1.0 - binary_entropy(torch.sigmoid(mask32))).clamp(0.0, 1.0)
        aux = {
            "noise_mask32": mask32,
            "noise_boundary32": boundary32,
            "noise_reliability32": reliability,
        }
        return noise_feat, reliability.reshape(b * k, 1, self.target_resolution, self.target_resolution), aux

    def apply_to_feature(self, noise_feat: torch.Tensor, main_feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        delta = self.delta_proj(noise_feat)
        gate = torch.sigmoid(self.gate(torch.cat([main_feat, noise_feat], dim=1)))
        if self.training and self.drop_path > 0.0:
            keep = torch.rand(delta.shape[0], 1, 1, 1, device=delta.device, dtype=delta.dtype) >= self.drop_path
            gate = gate * keep
        alpha = self.alpha()
        fused_delta = alpha.to(dtype=delta.dtype) * gate * delta
        debug = {
            "forensic_adapter_alpha": float(alpha.detach().cpu()),
            "forensic_adapter_gate_mean": float(gate.detach().float().mean().cpu()),
            "forensic_adapter_delta_norm": float(delta.detach().float().norm().cpu()),
        }
        return fused_delta, debug

    def forward(self, frames: torch.Tensor, main_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | dict[str, float]]]:
        noise_feat, reliability, aux = self.extract_features(frames, main_feat.dtype)
        fused_delta, debug = self.apply_to_feature(noise_feat, main_feat)
        debug["forensic_adapter_reliability_mean"] = float(reliability.detach().float().mean().cpu())
        return fused_delta, reliability, {**aux, "debug": debug}


class PrototypeMemoryState:
    def __init__(self, num_proto: int, dim: int, device: torch.device | None = None):
        device = device or torch.device("cpu")
        self.num_proto = int(num_proto)
        self.dim = int(dim)
        self.fg_proto = torch.zeros(self.num_proto, self.dim, device=device)
        self.bg_proto = torch.zeros(self.num_proto, self.dim, device=device)
        self.fg_valid = torch.zeros(self.num_proto, device=device)
        self.bg_valid = torch.zeros(self.num_proto, device=device)
        self.fg_ptr = 0
        self.bg_ptr = 0
        self.write_count = 0
        self.read_count = 0

    def to(self, device: torch.device) -> "PrototypeMemoryState":
        self.fg_proto = self.fg_proto.to(device)
        self.bg_proto = self.bg_proto.to(device)
        self.fg_valid = self.fg_valid.to(device)
        self.bg_valid = self.bg_valid.to(device)
        return self


class PrototypeMemory(nn.Module):
    """Foreground/background prototype memory with detached EMA writes."""

    def __init__(self, cfg: dict[str, Any] | None = None, channels: int = 128):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.channels = int(channels)
        self.num_fg_proto = int(cfg.get("num_fg_proto", 4))
        self.num_bg_proto = int(cfg.get("num_bg_proto", 4))
        self.num_proto = max(self.num_fg_proto, self.num_bg_proto)
        self.momentum = float(cfg.get("momentum", 0.90))
        self.temperature = float(cfg.get("temperature", 0.07))
        self.write_confidence_min = float(cfg.get("write_confidence_min", 0.75))
        self.write_entropy_max = float(cfg.get("write_entropy_max", 0.35))
        self.write_area_min = float(cfg.get("write_area_min", 0.002))
        self.write_area_max = float(cfg.get("write_area_max", 0.60))
        self.write_gate_min = float(cfg.get("write_gate_min", 0.15))
        self.write_warmup_epochs = int(cfg.get("write_warmup_epochs", 10))
        self.read_gate_max = float(cfg.get("read_gate_max", 0.60))
        read_bias = float(cfg.get("read_gate_bias_init", -1.5))

        self.q_proj = nn.Conv2d(self.channels, self.channels, 1)
        self.k_proj = nn.Linear(self.channels, self.channels)
        self.v_proj = nn.Linear(self.channels, self.channels)
        self.out_proj = nn.Conv2d(self.channels, self.channels, 1)
        self.read_gate = nn.Sequential(
            nn.Conv2d(self.channels * 2, max(16, self.channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(16, self.channels // 4), 1, 1),
        )
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.read_gate[-1].bias, read_bias)

    def new_state(self, device: torch.device | None = None) -> PrototypeMemoryState:
        return PrototypeMemoryState(self.num_proto, self.channels, device=device)

    def _state(self, bank, device: torch.device) -> PrototypeMemoryState:
        state = getattr(bank, "prototype_state", None)
        if state is None:
            state = self.new_state(device)
            setattr(bank, "prototype_state", state)
        return state.to(device)

    @staticmethod
    def _weighted_avg(feat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = weight.float().clamp_min(0.0)
        denom = weight.flatten(1).sum(dim=1).clamp_min(1.0e-6)
        value = (feat.float() * weight).flatten(2).sum(dim=2) / denom[:, None]
        return F.normalize(value, dim=1)

    def read(self, feat: torch.Tensor, bank) -> tuple[torch.Tensor, dict[str, float]]:
        state = self._state(bank, feat.device)
        valid = torch.cat([state.fg_valid[: self.num_fg_proto], state.bg_valid[: self.num_bg_proto]], dim=0) > 0
        if not bool(valid.any().detach().cpu()):
            zero = feat.sum() * 0.0
            for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.read_gate):
                for param in module.parameters():
                    if param.requires_grad:
                        zero = zero + param.sum() * 0.0
            return feat * 0.0 + zero, {
                "prototype_memory_read_gate_mean": 0.0,
                "prototype_memory_valid": 0.0,
                "prototype_memory_fg_proto_norm": 0.0,
                "prototype_memory_bg_proto_norm": 0.0,
            }
        proto = torch.cat([state.fg_proto[: self.num_fg_proto], state.bg_proto[: self.num_bg_proto]], dim=0)
        proto = proto[valid].to(device=feat.device, dtype=feat.dtype)
        b, c, h, w = feat.shape
        q = self.q_proj(feat).flatten(2).transpose(1, 2)
        k = self.k_proj(proto).transpose(0, 1)
        v = self.v_proj(proto)
        scale = max(self.temperature, 1.0e-4)
        attn = torch.softmax(torch.matmul(q, k) / scale, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        delta = self.out_proj(ctx)
        gate = torch.sigmoid(self.read_gate(torch.cat([feat, delta], dim=1))) * self.read_gate_max
        state.read_count += 1
        debug = {
            "prototype_memory_read_gate_mean": float(gate.detach().float().mean().cpu()),
            "prototype_memory_valid": float(valid.float().mean().detach().cpu()),
            "prototype_memory_fg_proto_norm": float(state.fg_proto.detach().float().norm().cpu()),
            "prototype_memory_bg_proto_norm": float(state.bg_proto.detach().float().norm().cpu()),
        }
        return gate * delta, debug

    def write(self, feat: torch.Tensor, prob: torch.Tensor, gate_mean: float, bank, epoch: int | None = None) -> dict[str, float]:
        state = self._state(bank, feat.device)
        prob = prob.detach().float()
        if prob.shape[-2:] != feat.shape[-2:]:
            prob = F.interpolate(prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        entropy = binary_entropy(prob).mean()
        confidence = 1.0 - entropy
        area = prob.mean()
        epoch_ready = epoch is None or int(epoch) >= self.write_warmup_epochs
        should_write = (
            epoch_ready
            and
            float(confidence.detach().cpu()) >= self.write_confidence_min
            and float(entropy.detach().cpu()) <= self.write_entropy_max
            and self.write_area_min <= float(area.detach().cpu()) <= self.write_area_max
            and float(gate_mean) >= self.write_gate_min
        )
        if not should_write:
            return {
                "prototype_memory_write_rate": 0.0,
                "prototype_memory_write_confidence": float(confidence.detach().cpu()),
                "prototype_memory_write_area": float(area.detach().cpu()),
                "prototype_memory_write_epoch_ready": float(1.0 if epoch_ready else 0.0),
            }

        fg_proto = self._weighted_avg(feat.detach(), prob)
        bg_proto = self._weighted_avg(feat.detach(), 1.0 - prob)
        fg = fg_proto.mean(dim=0)
        bg = bg_proto.mean(dim=0)
        fg_idx = state.fg_ptr % self.num_fg_proto
        bg_idx = state.bg_ptr % self.num_bg_proto
        with torch.no_grad():
            if state.fg_valid[fg_idx] > 0:
                state.fg_proto[fg_idx] = F.normalize(self.momentum * state.fg_proto[fg_idx] + (1.0 - self.momentum) * fg, dim=0)
            else:
                state.fg_proto[fg_idx] = fg
            if state.bg_valid[bg_idx] > 0:
                state.bg_proto[bg_idx] = F.normalize(self.momentum * state.bg_proto[bg_idx] + (1.0 - self.momentum) * bg, dim=0)
            else:
                state.bg_proto[bg_idx] = bg
            state.fg_valid[fg_idx] = 1.0
            state.bg_valid[bg_idx] = 1.0
            state.fg_ptr += 1
            state.bg_ptr += 1
            state.write_count += 1
        return {
            "prototype_memory_write_rate": 1.0,
            "prototype_memory_write_confidence": float(confidence.detach().cpu()),
            "prototype_memory_write_area": float(area.detach().cpu()),
            "prototype_memory_write_epoch_ready": float(1.0 if epoch_ready else 0.0),
        }
