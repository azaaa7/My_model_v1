from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_target(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1).detach()


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = int(kernel_size)

    @staticmethod
    def sobel(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
        target = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1]).float()
        pred_b = self.sobel(torch.sigmoid(logits))
        target_b = self.sobel(target)
        return F.l1_loss(pred_b, target_b)

