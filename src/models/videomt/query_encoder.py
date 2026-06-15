from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VideoMTQueryController(nn.Module):
    """Learned query propagation: q_in = Q_lrn or Linear(prev_q) + Q_lrn."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.dim = int(cfg.get("dim", 1024))
        self.num_queries = int(cfg.get("num_queries", 32))

        prop_cfg = cfg.get("propagation", {}) or {}
        self.detach_within_window = bool(prop_cfg.get("detach_within_window", False))
        self.detach_across_windows = bool(prop_cfg.get("detach_across_windows", True))

        qf_cfg = cfg.get("query_fusion", {}) or {}
        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.dim) * 0.02)
        self.prev_linear = nn.Linear(self.dim, self.dim, bias=bool(qf_cfg.get("linear_bias", True)))

        linear_init = str(qf_cfg.get("linear_init", "xavier_uniform"))
        if linear_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.prev_linear.weight)
            if self.prev_linear.bias is not None:
                nn.init.zeros_(self.prev_linear.bias)
        elif linear_init == "identity":
            nn.init.eye_(self.prev_linear.weight)
            if self.prev_linear.bias is not None:
                nn.init.zeros_(self.prev_linear.bias)

    def initial_queries(self, batch_size: int, device, dtype) -> torch.Tensor:
        return self.learned_queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def make_input_queries(
        self,
        batch_size: int,
        device,
        dtype,
        prev_q: torch.Tensor | None,
        detach_prev: bool = False,
    ) -> torch.Tensor:
        q_lrn = self.initial_queries(batch_size, device, dtype)
        if prev_q is None:
            return q_lrn
        if detach_prev:
            prev_q = prev_q.detach()
        return self.prev_linear(prev_q.to(device=device, dtype=dtype)) + q_lrn
