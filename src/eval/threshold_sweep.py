from __future__ import annotations

import torch

from .metrics import binary_metrics_from_logits


def sweep_thresholds(logits: torch.Tensor, masks: torch.Tensor, thresholds=None) -> tuple[float, dict[str, float]]:
    thresholds = thresholds or [i / 20 for i in range(1, 20)]
    best_t = 0.5
    best = {"iou": -1.0}
    for threshold in thresholds:
        metrics = binary_metrics_from_logits(logits, masks, threshold=threshold)
        if metrics["iou"] > best["iou"]:
            best_t = float(threshold)
            best = metrics
    return best_t, best

