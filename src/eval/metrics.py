from __future__ import annotations

import torch


def binary_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred = (torch.sigmoid(logits) > threshold).bool()
    gt = (target > threshold).bool()
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)

    tp = torch.logical_and(pred_flat, gt_flat).sum().float()
    fp = torch.logical_and(pred_flat, torch.logical_not(gt_flat)).sum().float()
    fn = torch.logical_and(torch.logical_not(pred_flat), gt_flat).sum().float()
    tn = torch.logical_and(torch.logical_not(pred_flat), torch.logical_not(gt_flat)).sum().float()

    union = tp + fp + fn
    iou = torch.where(union > 0, tp / (union + eps), torch.ones_like(union))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "iou": float(iou.detach().cpu()),
        "f1": float(f1.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

