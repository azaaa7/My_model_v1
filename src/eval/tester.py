from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import AverageMeter, binary_metrics_from_logits
from src.losses import AuxiliaryLoss, SegmentationLoss
from src.models import B23TFCUCCMFGMLiteModel
from src.train.checkpoint import load_checkpoint
from src.train.trainer import evaluate, make_loader
from src.utils.config import prepare_config


def _dataset_name(path: str) -> str:
    name = Path(path).stem
    for key in ("DVI", "CPNET", "OPN"):
        if key in name:
            return f"{key}_20" if "20" in name else key
    return name


def _sample_names(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [Path(str(item)).stem for item in value]
    if value:
        return [Path(str(value)).stem]
    return []


def _loss_summary(cfg: dict[str, Any]) -> str:
    parts = []
    for name, args in (cfg.get("loss", {}) or {}).items():
        weight = float((args or {}).get("weight", 1.0))
        if weight > 0:
            parts.append(f"{weight:g}*{name}")
    aux = []
    for name, args in (cfg.get("aux_loss", {}) or {}).items():
        if bool((args or {}).get("enabled", False)):
            aux.append(f"{float((args or {}).get('weight', 1.0)):g}*{name}")
    if aux:
        parts.append("aux(" + " + ".join(aux) + ")")
    return " + ".join(parts) if parts else "default"


def _ablation_summary(ablation: dict[str, Any] | None) -> str:
    ablation = ablation or {}
    enabled = [key for key in ("disable_ccm", "disable_fgm", "shuffle_bank", "zero_bank") if ablation.get(key)]
    return ", ".join(enabled) if enabled else "normal"


def print_test_summary(
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
    checkpoint: str,
    ablation: dict[str, Any] | None = None,
) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("                       TEST SUITE SUMMARY")
    lines.append("=" * 72)
    lines.append(f"  Checkpoint       : {checkpoint}")
    lines.append(f"  Model            : {cfg.get('model', {}).get('name', 'B23TFCUCCMFGMLiteModel')}")
    lines.append(f"  TFCU             : {cfg.get('tfcu', {}).get('version', '')}")
    lines.append(
        f"  LoRA             : rank={cfg.get('lora', {}).get('rank', '')} "
        f"alpha={cfg.get('lora', {}).get('alpha', '')}"
    )
    lines.append(
        f"  Temporal input   : num_clips={cfg.get('num_clips', '')} "
        f"num_frames={cfg.get('num_frames', '')} encoder_chunk={cfg.get('encoder_chunk', '')}"
    )
    lines.append(f"  Ablation         : {_ablation_summary(ablation)}")
    lines.append(f"  Loss             : {_loss_summary(cfg)}")
    lines.append(f"  Train datasets   : {', '.join(_sample_names(cfg.get('train_samples')))}")
    lines.append("-" * 72)
    lines.append(f"  {'Test Set':<14s} {'IoU':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'Loss':>8s}")
    lines.append("  " + "-" * 62)

    best_iou = -1.0
    best_name = ""
    for item in results:
        name = item["subset"]
        metrics = item["metrics"]
        lines.append(
            f"  {name:<14s} {metrics['iou']:8.4f} {metrics['f1']:8.4f} "
            f"{metrics['precision']:10.4f} {metrics['recall']:8.4f} {metrics['loss']:8.4f}"
        )
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_name = name

    lines.append("  " + "-" * 62)
    if results:
        avg = {
            key: sum(item["metrics"][key] for item in results) / len(results)
            for key in ("iou", "f1", "precision", "recall", "loss")
        }
        lines.append(
            f"  {'Average':<14s} {avg['iou']:8.4f} {avg['f1']:8.4f} "
            f"{avg['precision']:10.4f} {avg['recall']:8.4f} {avg['loss']:8.4f}"
        )
        lines.append(f"  Best: {best_name}  IoU={best_iou:.4f}")
    lines.append("=" * 72)
    print("\n".join(lines))


@torch.no_grad()
def evaluate_loader(model, loader, criterion, aux_criterion, device, cfg: dict[str, Any], ablation: dict[str, Any] | None = None):
    model.eval()
    return evaluate(model, loader, criterion, aux_criterion, device, cfg, ablation=ablation)


def run_test(cfg: dict[str, Any], checkpoint: str, ablation: dict[str, Any] | None = None):
    cfg = prepare_config(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = B23TFCUCCMFGMLiteModel(cfg).to(device)
    load_checkpoint(checkpoint, model, strict=False)
    criterion = SegmentationLoss(cfg.get("loss", {})).to(device)
    aux_criterion = AuxiliaryLoss(cfg.get("aux_loss", {})).to(device)

    results: list[dict[str, Any]] = []
    for sample_path in cfg.get("test_samples", []):
        run_cfg = dict(cfg)
        run_cfg["test_samples"] = [sample_path]
        loader = make_loader(run_cfg, "test", distributed=False)
        name = _dataset_name(sample_path)
        metrics = evaluate_loader(model, loader, criterion, aux_criterion, device, run_cfg, ablation=ablation)
        results.append({"subset": name, "metrics": metrics})
        print(
            f"[test/{name}] loss {metrics['loss']:.4f} "
            f"f1 {metrics['f1']:.4f} iou {metrics['iou']:.4f} "
            f"precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}"
        )
    print_test_summary(results, cfg, checkpoint, ablation=ablation)
    return results
