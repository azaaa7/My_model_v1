from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(model: nn.Module, cfg: dict[str, Any]):
    opt_cfg = cfg.get("optimizer", {}) or {}
    base_lr = float(opt_cfg.get("learning_rate", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    groups = {
        "lora": {"params": [], "lr": float(opt_cfg.get("lr_lora", 1e-5))},
        "temporal_encoder": {"params": [], "lr": float(opt_cfg.get("lr_temporal_encoder", base_lr))},
        "ccm": {"params": [], "lr": float(opt_cfg.get("lr_ccm", base_lr))},
        "fgm": {"params": [], "lr": float(opt_cfg.get("lr_fgm", base_lr))},
        "reliability_gate": {"params": [], "lr": float(opt_cfg.get("lr_reliability_gate", opt_cfg.get("lr_rgfgm", base_lr)))},
        "prototype_memory": {"params": [], "lr": float(opt_cfg.get("lr_prototype_memory", base_lr))},
        "forensic_adapter": {"params": [], "lr": float(opt_cfg.get("lr_forensic_adapter", opt_cfg.get("lr_noise_adapter", base_lr)))},
        "task_adapter": {"params": [], "lr": float(opt_cfg.get("lr_task_adapter", opt_cfg.get("lr_adapter", 1e-5)))},
        "tcu": {"params": [], "lr": float(opt_cfg.get("lr_tcu", base_lr))},
        "sumi": {"params": [], "lr": float(opt_cfg.get("lr_sumi", base_lr))},
        "forensic": {"params": [], "lr": float(opt_cfg.get("lr_forensic", base_lr))},
        "decoder": {"params": [], "lr": float(opt_cfg.get("lr_decoder", base_lr))},
        "query": {"params": [], "lr": float(opt_cfg.get("lr_query", base_lr))},
        "mask_head": {"params": [], "lr": float(opt_cfg.get("lr_mask_head", base_lr))},
        "other": {"params": [], "lr": base_lr},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name[len("module."):] if name.startswith("module.") else name
        if "lora_" in clean:
            groups["lora"]["params"].append(param)
        elif "query_controller." in clean:
            groups["query"]["params"].append(param)
        elif "query_mask_head." in clean:
            groups["mask_head"]["params"].append(param)
        elif "temporal_fusion" in clean or "temporal_encoder" in clean or "ttf" in clean:
            groups["temporal_encoder"]["params"].append(param)
        elif ".ccm." in clean:
            groups["ccm"]["params"].append(param)
        elif ".fgm." in clean:
            groups["fgm"]["params"].append(param)
        elif ".reliability_gate." in clean:
            groups["reliability_gate"]["params"].append(param)
        elif ".prototype_memory." in clean:
            groups["prototype_memory"]["params"].append(param)
        elif ".noise_adapter." in clean:
            groups["forensic_adapter"]["params"].append(param)
        elif ".task_adapter." in clean:
            groups["task_adapter"]["params"].append(param)
        elif ".tcu." in clean:
            groups["tcu"]["params"].append(param)
        elif ".sumi_heads." in clean:
            groups["sumi"]["params"].append(param)
        elif ".forensic_branch." in clean:
            groups["forensic"]["params"].append(param)
        elif "decoder." in clean or "feature_proj." in clean:
            groups["decoder"]["params"].append(param)
        else:
            groups["other"]["params"].append(param)

    param_groups = []
    for name, group in groups.items():
        if group["params"]:
            param_groups.append({"params": group["params"], "lr": group["lr"], "weight_decay": weight_decay, "name": name})
            print(f"[optimizer] group={name} params={sum(p.numel() for p in group['params'])} lr={group['lr']}")
    return AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1.0e-8)),
    )
