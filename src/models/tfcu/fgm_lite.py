from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadCrossAttention
from .cue_bank import ForgeryCueBank


class FeatureShiftPrompt(nn.Module):
    def __init__(self, cue_dim: int = 64, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cue_dim, int(hidden_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), cue_dim, kernel_size=1),
        )

    def forward(self, current_key: torch.Tensor, hist_cue: torch.Tensor) -> torch.Tensor:
        return hist_cue + self.net(current_key - hist_cue)


class FGMLite(nn.Module):
    """Historical forgery cue propagation and aggregation."""

    def __init__(self, in_channels: int = 1024, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.in_channels = int(in_channels)
        self.cue_dim = int(cfg.get("cue_dim", 64))
        self.cue_resolution = int(cfg.get("cue_resolution", 16))
        self.bank_len = int(cfg.get("bank_len", 3))
        self.detach_bank = bool(cfg.get("detach_bank", True))

        prop_cfg = cfg.get("propagation", {}) or {}
        agg_cfg = cfg.get("aggregation", {}) or {}
        prompt_cfg = cfg.get("prompt", {}) or {}
        self.propagation_enabled = bool(prop_cfg.get("enabled", True))
        self.aggregation_enabled = bool(agg_cfg.get("enabled", True))
        self.prompt_enabled = bool(prompt_cfg.get("enabled", True))
        self.hist_len = int(prop_cfg.get("hist_len", 2))
        self.prop_dim = int(prop_cfg.get("dim", 128))
        self.prop_heads = int(prop_cfg.get("heads", 4))
        self.use_topk = bool(prop_cfg.get("use_topk", True))
        self.topk = int(prop_cfg.get("topk", 128))
        self.diff_scale = float(agg_cfg.get("diff_scale", 1.0))
        self.use_center_frame = bool(agg_cfg.get("use_center_frame", True))
        self.use_first_last_diff = bool(agg_cfg.get("use_first_last_diff", True))

        self.norm = nn.GroupNorm(32, self.in_channels)
        self.q_proj = nn.Conv2d(self.in_channels, self.prop_dim, kernel_size=1)
        self.cue_to_prop = nn.Conv2d(self.cue_dim, self.prop_dim, kernel_size=1)
        self.prop_attn = MultiHeadCrossAttention(self.prop_dim, heads=self.prop_heads)
        self.prop_to_c = nn.Conv2d(self.prop_dim, self.in_channels, kernel_size=1)
        self.key_proj = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.Conv2d(self.in_channels, self.cue_dim, kernel_size=1),
        )
        self.ip_summary_proj = nn.Conv2d(self.in_channels, self.cue_dim, kernel_size=1)
        self.agg_attn = MultiHeadCrossAttention(self.cue_dim, heads=int(agg_cfg.get("heads", 4)))
        self.cue_to_ip = nn.Conv2d(self.cue_dim, self.in_channels, kernel_size=1)
        self.cue_feedback_scale = nn.Parameter(torch.tensor(float(agg_cfg.get("feedback_init", 0.01))))
        self.prompt = FeatureShiftPrompt(self.cue_dim, int(prompt_cfg.get("channels", 32)))
        self.aux_head = nn.Conv2d(self.in_channels, 1, kernel_size=1) if bool(cfg.get("aux_head", True)) else None
        if not self.prompt_enabled:
            for param in self.prompt.parameters():
                param.requires_grad = False
        if not self.propagation_enabled:
            for module in (self.norm, self.q_proj, self.cue_to_prop, self.prop_attn, self.prop_to_c):
                for param in module.parameters():
                    param.requires_grad = False
        if not self.aggregation_enabled:
            for module in (self.ip_summary_proj, self.agg_attn, self.cue_to_ip):
                for param in module.parameters():
                    param.requires_grad = False
            self.cue_feedback_scale.requires_grad = False

    @staticmethod
    def _zero_param_dependency(modules: tuple[nn.Module, ...], ref: torch.Tensor) -> torch.Tensor:
        dummy = None
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    value = param.sum() * 0.0
                    dummy = value if dummy is None else dummy + value
        if dummy is None:
            return ref.sum() * 0.0
        return dummy.to(device=ref.device, dtype=ref.dtype)

    def new_bank(
        self,
        *,
        shuffle_bank: bool = False,
        zero_bank: bool = False,
        detach_bank: bool | None = None,
    ) -> ForgeryCueBank:
        return ForgeryCueBank(
            bank_len=self.bank_len,
            detach_bank=self.detach_bank if detach_bank is None else bool(detach_bank),
            shuffle_bank=shuffle_bank,
            zero_bank=zero_bank,
        )

    def _current_key(self, f_cc: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = f_cc.shape
        if self.use_center_frame:
            key = f_cc[:, k // 2]
        else:
            key = f_cc.mean(dim=1)
        if self.use_first_last_diff and k > 1:
            key = key + self.diff_scale * (f_cc[:, -1] - f_cc[:, 0])
        key = F.adaptive_avg_pool2d(key, (self.cue_resolution, self.cue_resolution))
        return self.key_proj(key)

    def _propagate(self, f_cc: torch.Tensor, bank: ForgeryCueBank) -> tuple[torch.Tensor, dict[str, Any]]:
        b, k, c, h, w = f_cc.shape
        if not self.propagation_enabled or len(bank) == 0:
            zeros = torch.zeros(b, k, c, h, w, device=f_cc.device, dtype=f_cc.dtype)
            if self.propagation_enabled:
                modules = (self.norm, self.q_proj, self.cue_to_prop, self.prop_attn, self.prop_to_c)
                if self.prompt_enabled:
                    modules = (*modules, self.prompt)
                zeros = zeros + self._zero_param_dependency(modules, f_cc)
            return zeros, {"fgm_bank_size": len(bank), "fgm_propagated_norm": 0.0}

        q = self.q_proj(self.norm(f_cc.reshape(b * k, c, h, w)))
        q_tokens = q.reshape(b, k, self.prop_dim, h * w).permute(0, 1, 3, 2).reshape(b, k * h * w, self.prop_dim)

        current_key = self._current_key(f_cc)
        cues = bank.items(self.hist_len)
        prompted = []
        for cue in cues:
            if self.prompt_enabled:
                prompted.append(self.prompt(current_key, cue))
            else:
                prompted.append(cue)
        cue_stack = torch.stack(prompted, dim=1)
        flat_cues = cue_stack.reshape(b * len(prompted), self.cue_dim, self.cue_resolution, self.cue_resolution)
        kv = self.cue_to_prop(flat_cues).reshape(b, len(prompted), self.prop_dim, -1)
        kv_tokens = kv.permute(0, 1, 3, 2).reshape(b, len(prompted) * self.cue_resolution * self.cue_resolution, self.prop_dim)
        topk = self.topk if self.use_topk else None
        out, attn_debug = self.prop_attn(q_tokens, kv_tokens, topk=topk)
        prop = out.reshape(b, k, h, w, self.prop_dim).permute(0, 1, 4, 2, 3).reshape(b * k, self.prop_dim, h, w)
        f_ip = self.prop_to_c(prop).reshape(b, k, c, h, w)
        debug = {
            "fgm_bank_size": len(bank),
            "fgm_kv_cue_shape": tuple(kv_tokens.shape),
            "fgm_propagated_norm": float(f_ip.detach().float().norm().cpu()),
            **{f"fgm_prop_{k}": v for k, v in attn_debug.items()},
        }
        return f_ip, debug

    def _aggregate(self, f_cc: torch.Tensor, f_ip: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        current_key = self._current_key(f_cc)
        if not self.aggregation_enabled:
            return current_key, {"fgm_cue_norm": float(current_key.detach().float().norm().cpu())}

        b = f_cc.shape[0]
        ip_summary = f_ip.mean(dim=1)
        ip_summary = F.adaptive_avg_pool2d(ip_summary, (self.cue_resolution, self.cue_resolution))
        ip_summary = self.ip_summary_proj(ip_summary)
        tokens = torch.stack([current_key, ip_summary], dim=1)
        kv_tokens = tokens.reshape(b, 2, self.cue_dim, -1).permute(0, 1, 3, 2).reshape(
            b, 2 * self.cue_resolution * self.cue_resolution, self.cue_dim
        )
        q_tokens = current_key.flatten(2).transpose(1, 2)
        out, attn_debug = self.agg_attn(q_tokens, kv_tokens)
        cue = out.transpose(1, 2).reshape(b, self.cue_dim, self.cue_resolution, self.cue_resolution)
        debug = {
            "fgm_cue_shape": tuple(cue.shape),
            "fgm_cue_norm": float(cue.detach().float().norm().cpu()),
            **{f"fgm_agg_{k}": v for k, v in attn_debug.items()},
        }
        return cue, debug

    def forward(self, x: torch.Tensor, f_cc: torch.Tensor, bank: ForgeryCueBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if not self.enabled:
            b, k, c, h, w = x.shape
            f_ip = torch.zeros(b, k, c, h, w, device=x.device, dtype=x.dtype)
            cue = self._current_key(f_cc)
            return f_ip, cue, None, {"fgm_enabled": False, "fgm_bank_size": len(bank)}

        f_ip, prop_debug = self._propagate(f_cc, bank)
        cue, agg_debug = self._aggregate(f_cc, f_ip)
        if self.aggregation_enabled:
            cue_feedback = F.interpolate(cue, size=f_ip.shape[-2:], mode="bilinear", align_corners=False)
            cue_feedback = self.cue_to_ip(cue_feedback)[:, None]
            f_ip = f_ip + self.cue_feedback_scale * cue_feedback
            agg_debug["fgm_cue_feedback_norm"] = float(cue_feedback.detach().float().norm().cpu())
            agg_debug["fgm_cue_feedback_scale"] = float(self.cue_feedback_scale.detach().cpu())
        aux = self.aux_head(f_ip.reshape(-1, f_ip.shape[2], f_ip.shape[3], f_ip.shape[4])).reshape(
            f_ip.shape[0], f_ip.shape[1], 1, f_ip.shape[3], f_ip.shape[4]
        ) if self.aux_head else None
        debug = {"fgm_enabled": True, **prop_debug, **agg_debug}
        return f_ip, cue, aux, debug
