from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_tensor_with_offset(x: torch.Tensor, offset: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    b, _, h, w = x.shape
    ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
    norm = torch.empty_like(offset)
    norm[:, 0] = offset[:, 0] * 2.0 / max(w - 1, 1)
    norm[:, 1] = offset[:, 1] * 2.0 / max(h - 1, 1)
    grid = base + norm.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)

