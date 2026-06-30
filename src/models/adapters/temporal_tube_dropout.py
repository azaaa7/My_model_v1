from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTubeDropout(nn.Module):
    """Drop the same spatial tube across time during training only."""

    def __init__(self, drop_prob: float = 0.10) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TemporalTubeDropout expects [G,K,C,H,W], got {tuple(x.shape)}")
        if (not self.training) or self.drop_prob <= 0.0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, x.shape[-2], x.shape[-1], device=x.device) > self.drop_prob).to(x.dtype)
        return x * keep / max(1.0 - self.drop_prob, 1.0e-6)
