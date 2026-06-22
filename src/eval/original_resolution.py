from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import _align_mask_to_frame, _load_samples, _numeric_key
from src.data.transforms import is_image_file, threshold_mask
from src.eval.metrics import AverageMeter, binary_metrics_from_logits


def make_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def make_hann_weight(tile_size: int, min_weight: float = 1.0e-3, device=None) -> torch.Tensor:
    win = torch.hann_window(tile_size, periodic=False, device=device, dtype=torch.float32)
    weight = torch.outer(win, win).clamp_min(float(min_weight))
    return weight.view(1, 1, tile_size, tile_size)


def _pad_clip(clip: torch.Tensor, tile_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = clip.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h or pad_w:
        clip = F.pad(clip, (0, pad_w, 0, pad_h), mode="replicate")
    return clip, h, w


@torch.no_grad()
def tiled_clip_logits(
    model,
    clip: torch.Tensor,
    cfg: dict[str, Any],
    ablation: dict[str, Any] | None = None,
) -> torch.Tensor:
    if clip.ndim != 4:
        raise ValueError(f"clip must be [T,3,H,W], got {tuple(clip.shape)}")
    infer_cfg = cfg.get("inference", {}) or {}
    tile_size = int(infer_cfg.get("tile_size", 512))
    stride = int(infer_cfg.get("tile_stride", 384))
    min_weight = float(infer_cfg.get("hann_min_weight", 1.0e-3))
    clip, orig_h, orig_w = _pad_clip(clip, tile_size)
    _, _, padded_h, padded_w = clip.shape
    y_starts = make_tile_starts(padded_h, tile_size, stride)
    x_starts = make_tile_starts(padded_w, tile_size, stride)
    acc = torch.zeros(clip.shape[0], 1, padded_h, padded_w, device=clip.device, dtype=torch.float32)
    weight_acc = torch.zeros_like(acc)
    weight = make_hann_weight(tile_size, min_weight=min_weight, device=clip.device)

    for y0 in y_starts:
        for x0 in x_starts:
            tile = clip[:, :, y0:y0 + tile_size, x0:x0 + tile_size]
            out = model(tile.unsqueeze(0).unsqueeze(0), mode="eval", ablation=ablation)
            logits = out["logits"][0, 0].float()
            acc[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += logits * weight
            weight_acc[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += weight
    logits = acc / weight_acc.clamp_min(1.0e-6)
    return logits[:, :, :orig_h, :orig_w]


def _read_video_and_masks(video_dir: str, mask_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    frame_list = sorted([p for p in os.listdir(video_dir) if is_image_file(p)], key=_numeric_key)
    mask_list = sorted([p for p in os.listdir(mask_dir) if is_image_file(p)], key=_numeric_key)
    if len(frame_list) != len(mask_list):
        raise ValueError(f"Frame count and mask count mismatch in {video_dir}: {len(frame_list)} vs {len(mask_list)}")
    frames = []
    masks = []
    for frame_name, mask_name in zip(frame_list, mask_list):
        frame_bgr = cv2.imread(str(Path(video_dir) / frame_name), cv2.IMREAD_COLOR)
        mask_bgr = cv2.imread(str(Path(mask_dir) / mask_name), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {Path(video_dir) / frame_name}")
        if mask_bgr is None:
            raise FileNotFoundError(f"Failed to read mask: {Path(mask_dir) / mask_name}")
        mask_bgr = _align_mask_to_frame(mask_bgr, frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = threshold_mask(mask_bgr).astype(np.float32)[:, :, :1] / 255.0
        frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
        masks.append(torch.from_numpy(mask).permute(2, 0, 1))
    return torch.stack(frames, dim=0), torch.stack(masks, dim=0)


def _clip_indices(num_frames_total: int, clip_len: int, stride: int) -> list[tuple[list[int], list[bool]]]:
    starts = list(range(0, num_frames_total, max(1, stride)))
    clips = []
    for start in starts:
        indices = [min(start + i, num_frames_total - 1) for i in range(clip_len)]
        valid = [start + i < num_frames_total for i in range(clip_len)]
        clips.append((indices, valid))
    return clips


@torch.no_grad()
def evaluate_original_resolution(
    model,
    criterion,
    aux_criterion,
    sumi_criterion,
    device,
    cfg: dict[str, Any],
    ablation: dict[str, Any] | None = None,
    epoch: int = 10**9,
    include_aux_losses: bool = False,
) -> dict[str, float]:
    del aux_criterion, sumi_criterion, include_aux_losses
    model.eval()
    meters = {key: AverageMeter() for key in ["loss", "iou", "f1", "precision", "recall", "accuracy"]}
    sample_paths = cfg.get("test_samples") if cfg.get("type") == "test" else cfg.get("val_samples")
    if not sample_paths:
        sample_paths = cfg.get("test_samples") or cfg.get("val_samples")
    samples = _load_samples(sample_paths)
    clip_len = int(cfg.get("num_frames", 5))
    clip_stride = int((cfg.get("inference", {}) or {}).get("clip_stride", cfg.get("clip_stride", 1)))

    for video_dir, mask_dir in samples:
        frames_cpu, masks_cpu = _read_video_and_masks(video_dir, mask_dir)
        num_total = int(frames_cpu.shape[0])
        h, w = int(frames_cpu.shape[-2]), int(frames_cpu.shape[-1])
        logit_acc = torch.zeros(num_total, 1, h, w, device=device, dtype=torch.float32)
        count_acc = torch.zeros(num_total, 1, 1, 1, device=device, dtype=torch.float32)
        for indices, valid in _clip_indices(num_total, clip_len, clip_stride):
            clip = frames_cpu[indices].to(device=device, non_blocking=True)
            logits_clip = tiled_clip_logits(model, clip, cfg, ablation=ablation)
            for local_idx, frame_idx in enumerate(indices):
                if not valid[local_idx]:
                    continue
                logit_acc[frame_idx] += logits_clip[local_idx]
                count_acc[frame_idx] += 1.0
        logits = logit_acc / count_acc.clamp_min(1.0)
        target = masks_cpu.to(device=device, non_blocking=True)
        try:
            total_loss, _loss_items = criterion(logits, target, aux=None, epoch=epoch, include_aux=False)
        except TypeError:
            total_loss, _loss_items = criterion(logits, target)
        metrics = binary_metrics_from_logits(logits, target)
        weight = num_total
        meters["loss"].update(float(total_loss.detach().cpu()), weight)
        for key, value in metrics.items():
            meters[key].update(value, weight)
    return {key: meter.avg for key, meter in meters.items()}
