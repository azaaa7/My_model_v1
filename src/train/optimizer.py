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
        "ccm": {"params": [], "lr": float(opt_cfg.get("lr_ccm", base_lr))},
        "fgm": {"params": [], "lr": float(opt_cfg.get("lr_fgm", base_lr))},
        "forensic": {"params": [], "lr": float(opt_cfg.get("lr_forensic", base_lr))},
        "decoder": {"params": [], "lr": float(opt_cfg.get("lr_decoder", base_lr))},
        "other": {"params": [], "lr": base_lr},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"]["params"].append(param)
        elif ".ccm." in name:
            groups["ccm"]["params"].append(param)
        elif ".fgm." in name:
            groups["fgm"]["params"].append(param)
        elif ".forensic_branch." in name:
            groups["forensic"]["params"].append(param)
        elif ".decoder." in name:
            groups["decoder"]["params"].append(param)
        else:
            groups["other"]["params"].append(param)

    param_groups = []
    for name, group in groups.items():
        if group["params"]:
            param_groups.append({"params": group["params"], "lr": group["lr"], "weight_decay": weight_decay, "name": name})
            print(f"[optimizer] group={name} params={sum(p.numel() for p in group['params'])} lr={group['lr']}")
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
