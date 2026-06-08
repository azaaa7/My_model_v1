from __future__ import annotations

from typing import Any

import torch


def tensor_stats(x: torch.Tensor | None) -> dict[str, Any]:
    if x is None:
        return {}
    with torch.no_grad():
        return {
            "shape": tuple(x.shape),
            "mean": float(x.detach().float().mean().cpu()),
            "std": float(x.detach().float().std().cpu()),
            "norm": float(x.detach().float().norm().cpu()),
        }

