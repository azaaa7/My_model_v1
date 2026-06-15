from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.data import build_dataloader
from src.eval.metrics import AverageMeter, binary_metrics_from_logits
from src.losses import AuxiliaryLoss, CompositeForensicLoss, SegmentationLoss, SUMILocalizationLoss, TTFMinimalLoss, VideoMTLoss, VideoMTQueryMaskLoss
from src.models.builder import build_model
from src.models.b23_tfcu_ccm_fgm_model import count_parameters, count_trainable_by_keyword
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.optimizer import build_optimizer
from src.train.scheduler import build_scheduler
from src.utils.config import prepare_config
from src.utils.distributed import is_main_process
from src.utils.logger import dump_json, ensure_dir, log_debug_dict


METRIC_KEYS = ["iou", "f1", "precision", "recall", "accuracy"]
DEBUG_METRIC_PREFIXES = (
    "reliability_gate_",
    "forensic_adapter_",
    "prototype_memory_",
    "adapter_",
    "tcu_",
    "temporal_encoder_",
    "ttf_",
    "videomt_",
    "sumi_",
    "background_suppression_",
)
ALWAYS_LOG_METRICS = (
    "adapter_mask32_loss",
    "adapter_boundary32_loss",
    "tcu_momentary_loss",
    "tcu_gradual_loss",
    "tcu_cumulative_loss",
    "sumi_sufficiency_loss",
    "sumi_ib_kl",
    "sumi_source_adv_loss",
    "background_suppression_loss",
    "sumi_loss",
    "adapter_alpha",
    "adapter_gate_mean",
    "tcu_alpha",
    "tcu_gate_momentary_mean",
    "tcu_gate_gradual_mean",
    "tcu_gate_cumulative_mean",
    "tcu_quality_mean",
    "ttf_alpha",
    "ttf_residual_energy",
    "loss_total",
    "loss_seg",
    "loss_bce",
    "loss_focal_bce",
    "loss_dice",
    "loss_edge",
    "loss_query_bce",
    "loss_query_dice",
    "loss_query_no_object",
    "loss_query_weight_scale",
    "loss_tda",
    "tda_weight",
)


def _format_log_value(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def _current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def build_loss(cfg: dict[str, Any]):
    loss_cfg = cfg.get("loss", {}) or {}
    loss_type = str(loss_cfg.get("type", "")).lower()
    loss_name = str(loss_cfg.get("name", "")).lower()
    if loss_type == "composite_forensic":
        return CompositeForensicLoss(loss_cfg), None
    if loss_type == "ttf_minimal":
        return TTFMinimalLoss(loss_cfg), None
    if loss_type == "videomt_query_mask":
        return VideoMTQueryMaskLoss(loss_cfg), None
    if loss_type == "videomt" or loss_name == "videomtloss":
        return VideoMTLoss(loss_cfg), None
    return SegmentationLoss(loss_cfg), AuxiliaryLoss(cfg.get("aux_loss", {}))


def _update_meter(meters: dict[str, AverageMeter], key: str, value: float, n: int = 1) -> None:
    if key not in meters:
        meters[key] = AverageMeter()
    meters[key].update(float(value), n)


def _meters_to_dict(meters: dict[str, AverageMeter]) -> dict[str, float]:
    return {key: meter.avg for key, meter in meters.items()}


def _sample_names(value) -> str:
    if isinstance(value, (list, tuple)):
        return ";".join(Path(str(item)).name for item in value)
    return Path(str(value)).name if value else ""


def _log_fields(train_metrics: dict[str, float], val_metrics: dict[str, float] | None) -> list[str]:
    fields = ["epoch", "lr"]
    train_keys = ["loss", "main_loss", "aux_loss"]
    train_keys.extend(key for key in sorted(train_metrics) if key.endswith("_loss") and key not in train_keys)
    train_keys.extend(key for key in ALWAYS_LOG_METRICS if key not in train_keys)
    train_keys.extend(METRIC_KEYS)
    train_keys.extend(
        key
        for key in sorted(train_metrics)
        if key not in train_keys and key not in METRIC_KEYS and any(key.startswith(prefix) for prefix in DEBUG_METRIC_PREFIXES)
    )
    fields.extend(f"train_{key}" for key in train_keys)

    val_keys = ["loss", "main_loss", "aux_loss"]
    val_metric_keys = set(train_metrics)
    if val_metrics:
        val_metric_keys.update(val_metrics)
    val_keys.extend(key for key in sorted(val_metric_keys) if key.endswith("_loss") and key not in val_keys)
    val_keys.extend(key for key in ALWAYS_LOG_METRICS if key not in val_keys)
    val_keys.extend(METRIC_KEYS)
    val_keys.extend(
        key
        for key in sorted(val_metric_keys)
        if key not in val_keys and key not in METRIC_KEYS and any(key.startswith(prefix) for prefix in DEBUG_METRIC_PREFIXES)
    )
    fields.extend(f"val_{key}" for key in val_keys)
    return fields


def _debug_scalar_items(debug: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(debug, dict):
        return {}
    items: dict[str, float] = {}

    def visit(value):
        if isinstance(value, dict):
            for key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and any(str(key).startswith(prefix) for prefix in DEBUG_METRIC_PREFIXES):
                    items[str(key)] = float(sub_value)
                elif isinstance(sub_value, dict):
                    visit(sub_value)

    visit(debug)
    return items


def _init_log_txt(path: Path, cfg: dict[str, Any], model) -> None:
    expected_prefix = (
        f"# model={cfg.get('model', {}).get('name', 'B23TFCUCCMFGMLiteModel')} "
        f"tfcu={cfg.get('tfcu', {}).get('version', '')} "
        f"lora_rank={cfg.get('lora', {}).get('rank', '')} input_size={cfg.get('input_size', '')} "
        f"num_clips={cfg.get('num_clips', '')} num_frames={cfg.get('num_frames', '')}"
    )
    if path.exists():
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first_line and first_line[0] == expected_prefix:
            return
        path.replace(path.with_suffix(path.suffix + ".old"))
    total, trainable = count_parameters(model.module if hasattr(model, "module") else model)
    loss_cfg = cfg.get("loss", {}) or {}
    aux_cfg = cfg.get("aux_loss", {}) or {}
    loss_parts = []
    for name, args in loss_cfg.items():
        if name == "type":
            loss_parts.append(str(args))
            continue
        if not isinstance(args, dict):
            continue
        if args.get("enabled") is False:
            continue
        weight = float((args or {}).get("weight", 1.0))
        if weight > 0:
            loss_parts.append(f"{weight:g}*{name}")
    aux_parts = []
    for name, args in aux_cfg.items():
        if not isinstance(args, dict):
            continue
        if bool((args or {}).get("enabled", False)):
            aux_parts.append(f"{float((args or {}).get('weight', 1.0)):g}*{name}")
    lines = [
        expected_prefix,
        f"# loss={' + '.join(loss_parts)} aux={' + '.join(aux_parts)}",
        f"# train_samples={_sample_names(cfg.get('train_samples'))} val_samples={_sample_names(cfg.get('val_samples'))}",
        f"# batch_size={cfg.get('batch_size', '')} lr={cfg.get('optimizer', {}).get('learning_rate', '')} "
        f"params_total={total} params_trainable={trainable}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_epoch_log(path: Path, epoch: int, lr: float, train_metrics: dict[str, float], val_metrics: dict[str, float] | None) -> None:
    fields = _log_fields(train_metrics, val_metrics)
    has_header = False
    header_matches = False
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("epoch,"):
                    has_header = True
                    header_matches = line.rstrip("\n").split(",") == fields
                    break
        if has_header and not header_matches:
            path.replace(path.with_suffix(path.suffix + ".bad_header.bak"))
            has_header = False
    if not has_header:
        with open(path, "a", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")

    row: dict[str, float | int | None] = {"epoch": epoch, "lr": lr}
    for key, value in train_metrics.items():
        row[f"train_{key}"] = value
    if val_metrics is not None:
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value

    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(_format_log_value(row.get(field)) for field in fields) + "\n")


def make_loader(cfg: dict[str, Any], mode: str, distributed: bool = False, rank: int = 0, world_size: int = 1):
    samples_key = "test_samples" if mode == "test" else "val_samples" if mode == "val" else "train_samples"
    bank_cfg = cfg.get("fgm_bank", {}) or {}
    num_clips = int(cfg.get(f"{mode}_num_clips", cfg.get("num_clips", 4)))
    test_max_clips = int(cfg.get(f"{mode}_test_max_clips", cfg.get("test_max_clips", num_clips)))
    num_workers = int(cfg.get(f"{mode}_num_workers", cfg.get("num_workers", 4)))
    return build_dataloader(
        samples=cfg[samples_key],
        mode=mode,
        batch_size=int(cfg.get("batch_size", 1)) if mode == "train" else 1,
        num_workers=num_workers,
        input_size=int(cfg.get("input_size", 512)),
        gt_ratio=int(cfg.get("gt_ratio", 1)),
        num_frames=int(cfg.get("num_frames", 4)),
        dataset_repeat=int(cfg.get("dataset_repeat", 1)),
        augment_prob=float(cfg.get("augment_prob", 0.75)),
        spatial_augment_prob=cfg.get("spatial_augment_prob", None),
        appearance_augment_prob=cfg.get("appearance_augment_prob", None),
        num_clips=num_clips,
        clip_stride=int(cfg.get("clip_stride", 1)),
        use_tfcu_adapter=True,
        test_max_clips=test_max_clips,
        train_full_video_windows=bool(bank_cfg.get("train_full_video_windows", cfg.get("train_full_video_windows", False))),
        train_max_windows_per_video=int(cfg.get("train_max_windows_per_video", 0) or 0),
        val_full_video=bool(cfg.get("val_full_video", False)),
        test_full_video=bool(cfg.get("test_full_video", True)),
        temporal_augment=cfg.get("temporal_augment", {}) or {},
        seed=int(cfg.get("seed", 0)),
        robust_noise_snr=int(cfg.get("robust_noise_snr", 0)),
        robust_jpeg_quality=int(cfg.get("robust_jpeg_quality", 0)),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["images"], batch["masks"], batch.get("name", batch.get("video_id", "sample"))
    images, masks, _oh, _ow, name = batch
    return images, masks, name


def _inner_model(model):
    return model.module if hasattr(model, "module") else model


def _supports_fgm_bank(model) -> bool:
    return hasattr(_inner_model(model), "new_fgm_bank")


def _new_fgm_bank(model, ablation: dict[str, Any] | None = None):
    return _inner_model(model).new_fgm_bank(ablation)


def _stateful_bank_enabled(cfg: dict[str, Any], mode: str) -> bool:
    model_name = str((cfg.get("model", {}) or {}).get("name", "B23TFCUCCMFGMLiteModel"))
    if model_name == "B23VideoMTWindowModel":
        return False
    bank_cfg = cfg.get("fgm_bank", {}) or {}
    if mode == "train":
        return bool(bank_cfg.get("stateful_train", bank_cfg.get("train_full_video_windows", False)))
    default_eval = bool(cfg.get("val_full_video", False) or cfg.get("test_full_video", False))
    return bool(bank_cfg.get("stateful_eval", default_eval))


def _is_videomt_model(cfg: dict[str, Any]) -> bool:
    return str((cfg.get("model", {}) or {}).get("name", "")) == "B23VideoMTWindowModel"


def _videomt_stateful_enabled(cfg: dict[str, Any], mode: str) -> bool:
    if not _is_videomt_model(cfg):
        return False
    prop_cfg = ((cfg.get("videomt", {}) or {}).get("propagation", {}) or {})
    if mode == "train":
        return bool(prop_cfg.get("stateful_windows", False))
    return bool(prop_cfg.get("carry_state_in_eval", True))


def _reset_bank_on_new_video(cfg: dict[str, Any]) -> bool:
    bank_cfg = cfg.get("fgm_bank", {}) or {}
    return bool(bank_cfg.get("reset_on_new_video", True))


def _batch_first(value, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _batch_str(batch: dict[str, Any], key: str, default: str = "") -> str:
    return str(_batch_first(batch.get(key), default))


def _batch_bool(batch: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(_batch_first(batch.get(key), default))


def _has_videomt_state_meta(batch) -> bool:
    return (
        isinstance(batch, dict)
        and "video_id" in batch
        and "is_first_window" in batch
        and "is_last_window" in batch
    )


def _valid_mask_from_batch(batch, device) -> torch.Tensor | None:
    if not isinstance(batch, dict) or "valid_mask" not in batch:
        return None
    valid = batch["valid_mask"]
    if not torch.is_tensor(valid):
        valid = torch.as_tensor(valid)
    return valid.to(device=device, non_blocking=True).bool()


def _filter_by_valid_mask(logits: torch.Tensor, target: torch.Tensor, aux: dict[str, Any], valid_mask: torch.Tensor | None):
    if valid_mask is None or bool(valid_mask.all().detach().cpu()):
        return logits, target, aux
    valid = valid_mask.reshape(-1)

    def filter_tensor(x: torch.Tensor | None):
        if x is None or not torch.is_tensor(x):
            return x
        if x.ndim >= valid_mask.ndim + 1 and tuple(x.shape[:valid_mask.ndim]) == tuple(valid_mask.shape):
            return x.reshape(-1, *x.shape[valid_mask.ndim:])[valid]
        if x.ndim >= valid_mask.ndim + 3 and tuple(x.shape[:valid_mask.ndim]) == tuple(valid_mask.shape):
            return x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])[valid]
        return x

    filtered_aux: dict[str, Any] = {}
    for key, value in aux.items():
        if key == "debug":
            filtered_aux[key] = value
        else:
            filtered_aux[key] = filter_tensor(value)
    return filter_tensor(logits), filter_tensor(target), filtered_aux


def _num_target_frames(target: torch.Tensor) -> int:
    if target.ndim < 4:
        return int(target.shape[0]) if target.ndim else 1
    return int(target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1]).shape[0])


def _distributed_any(flag: bool, device: torch.device) -> bool:
    if dist.is_available() and dist.is_initialized():
        value = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())
    return bool(flag)


def _distributed_barrier(device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend().lower() == "nccl" and device.type == "cuda":
        current = device.index if device.index is not None else torch.cuda.current_device()
        dist.barrier(device_ids=[current])
    else:
        dist.barrier()


def _grad_norm_and_finite(params: list[torch.nn.Parameter], max_norm: float) -> tuple[torch.Tensor, bool]:
    if max_norm > 0:
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, error_if_nonfinite=False)
    else:
        norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
        norm = torch.stack(norms).norm() if norms else torch.tensor(0.0)
    return norm, bool(torch.isfinite(norm.detach()).item())


def _nonfinite_state_names(model, limit: int = 8) -> list[str]:
    bad: list[str] = []
    for name, tensor in _inner_model(model).state_dict().items():
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def align_logits_masks(logits: torch.Tensor, masks: torch.Tensor):
    if masks.ndim == 4:
        # Single-clip train mode returns only the center-frame mask:
        # logits [B,M,K,1,H,W] or [B,K,1,H,W] -> [B,1,H,W].
        if logits.ndim == 6:
            logits = logits[:, logits.shape[1] // 2, logits.shape[2] // 2]
        elif logits.ndim == 5:
            logits = logits[:, logits.shape[1] // 2]
    elif masks.ndim == 5:
        # Single-clip val/test returns all frame masks [B,K,1,H,W].
        # Model output is [B,1,K,1,H,W], so add the clip dimension.
        masks = masks[:, None]
    if logits.shape[-2:] != masks.shape[-2:]:
        flat = logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1])
        flat = F.interpolate(flat, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        logits = flat.reshape(*logits.shape[:-2], masks.shape[-2], masks.shape[-1])
    return logits, masks


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    aux_criterion,
    sumi_criterion,
    device,
    cfg,
    ablation: dict[str, Any] | None = None,
    epoch: int = 10**9,
    include_aux_losses: bool = True,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    meters = {key: AverageMeter() for key in ["loss", "iou", "f1", "precision", "recall", "accuracy"]}
    supports_fgm_bank = _supports_fgm_bank(model)
    use_stateful_bank = supports_fgm_bank and _stateful_bank_enabled(cfg, "eval")
    use_videomt_stateful = _videomt_stateful_enabled(cfg, "eval")
    reset_on_new_video = _reset_bank_on_new_video(cfg)
    fgm_bank = None
    videomt_state = None
    current_video_id = None
    for batch in loader:
        images, masks, _name = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if use_stateful_bank and isinstance(batch, dict):
            video_id = _batch_str(batch, "video_id", _name)
            should_reset = fgm_bank is None
            if reset_on_new_video:
                should_reset = should_reset or video_id != current_video_id or _batch_bool(batch, "is_first_window", False)
            if should_reset:
                fgm_bank = _new_fgm_bank(model, ablation)
                current_video_id = video_id
        else:
            fgm_bank = None
        can_use_videomt_state = use_videomt_stateful and _has_videomt_state_meta(batch)
        if can_use_videomt_state:
            video_id = _batch_str(batch, "video_id", _name)
            should_reset = videomt_state is None or video_id != current_video_id or _batch_bool(batch, "is_first_window", False)
            if should_reset:
                videomt_state = None
                current_video_id = video_id
        else:
            videomt_state = None

        if _is_videomt_model(cfg):
            out = model(
                images,
                mode="eval",
                ablation=ablation,
                videomt_state=videomt_state,
                return_videomt_state=can_use_videomt_state,
            )
        elif supports_fgm_bank:
            out = model(images, mode="eval", ablation=ablation, fgm_bank=fgm_bank, return_fgm_bank=use_stateful_bank)
        else:
            out = model(images, mode="eval", ablation=ablation)
        if use_stateful_bank:
            fgm_bank = out.get("fgm_bank", fgm_bank)
        if can_use_videomt_state:
            videomt_state = out.get("videomt_state", videomt_state)
        logits, target = align_logits_masks(out["logits"], masks)
        valid_mask = _valid_mask_from_batch(batch, device)
        logits, target, aux = _filter_by_valid_mask(logits, target, out["aux"], valid_mask)
        if aux_criterion is None:
            total_loss, loss_items = criterion(
                logits,
                target,
                aux=aux,
                epoch=epoch,
                include_aux=include_aux_losses,
            )
            sumi_items = {}
        else:
            loss, _ = criterion(logits, target)
            if include_aux_losses:
                aux_loss, _ = aux_criterion(aux, target)
                sumi_loss, sumi_items = sumi_criterion(aux, target, epoch=epoch, source_names=_name)
            else:
                aux_loss = target.sum() * 0.0
                sumi_loss = target.sum() * 0.0
                sumi_items = {}
            total_loss = loss + aux_loss + sumi_loss
            loss_items = {}
        metrics = binary_metrics_from_logits(
            logits.reshape(-1, 1, logits.shape[-2], logits.shape[-1]),
            target.reshape(-1, 1, target.shape[-2], target.shape[-1]),
        )
        batch_weight = _num_target_frames(target)
        meters["loss"].update(float(total_loss.detach().cpu()), batch_weight)
        for key, value in metrics.items():
            meters[key].update(value, batch_weight)
        for key, value in _debug_scalar_items(aux.get("debug", {})).items():
            _update_meter(meters, key, value, batch_weight)
        for key, value in loss_items.items():
            _update_meter(meters, key, value, batch_weight)
        for key, value in sumi_items.items():
            _update_meter(meters, key, value, batch_weight)
        if reset_on_new_video and use_stateful_bank and isinstance(batch, dict) and _batch_bool(batch, "is_last_window", False):
            fgm_bank = None
            current_video_id = None
        if can_use_videomt_state and _batch_bool(batch, "is_last_window", False):
            videomt_state = None
            current_video_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {key: meter.avg for key, meter in meters.items()}


def _print_param_report(model) -> None:
    total, trainable = count_parameters(model)
    print(f"total params: {total}")
    print(f"trainable params: {trainable}")
    print(f"LoRA trainable params: {count_trainable_by_keyword(model, ('lora_',))}")
    print(f"TFCU trainable params: {count_trainable_by_keyword(model, ('ccm.', 'fgm.', 'fusion.', 'static_adapter.'))}")
    print(f"Task adapter trainable params: {count_trainable_by_keyword(model, ('task_adapter.',))}")
    print(f"TCU unravel trainable params: {count_trainable_by_keyword(model, ('tcu.',))}")
    print(f"TTF temporal encoder trainable params: {count_trainable_by_keyword(model, ('temporal_fusion.', 'ttf'))}")
    print(
        "VidEoMT query trainable params: "
        f"{count_trainable_by_keyword(model, ('query_fusion.', 'query_controller.', 'query_mask_head.', 'feature_proj.'))}"
    )
    print(f"SUMI heads trainable params: {count_trainable_by_keyword(model, ('sumi_heads.',))}")
    print(f"decoder trainable params: {count_trainable_by_keyword(model, ('decoder.',))}")


def run_train(cfg: dict[str, Any], distributed: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    cfg = prepare_config(cfg)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    save_dir = ensure_dir((cfg.get("train", {}) or {}).get("save_dir", "runs/b23_ccm_fgm_lite_lora32"))
    log_txt = Path(save_dir) / "log.txt"
    if is_main_process():
        dump_json(Path(save_dir) / "config.resolved.json", cfg)

    train_loader = make_loader(cfg, "train", distributed=distributed, rank=rank, world_size=world_size)
    val_loader = make_loader(cfg, "val", distributed=distributed, rank=rank, world_size=world_size)
    main_val_loader = None
    if distributed and is_main_process():
        # Full-video/stateful validation has uneven video-window counts per rank.
        # Rank 0 runs complete validation on the unwrapped model while others wait.
        main_val_loader = make_loader(cfg, "val", distributed=False)
    model = build_model(cfg).to(device)
    if is_main_process():
        _print_param_report(model)
        _init_log_txt(log_txt, cfg, model)
    find_unused = bool((cfg.get("ddp", {}) or {}).get("find_unused_parameters", False))
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion, aux_criterion = build_loss(cfg)
    criterion = criterion.to(device)
    if aux_criterion is not None:
        aux_criterion = aux_criterion.to(device)
        sumi_criterion = SUMILocalizationLoss((cfg.get("loss", {}) or {}).get("sumi", {})).to(device)
    else:
        sumi_criterion = None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and torch.cuda.is_available())
    start_epoch = 0
    best_iou = -1.0
    if cfg.get("checkpoint"):
        ckpt = load_checkpoint(cfg["checkpoint"], model, optimizer=optimizer, scheduler=scheduler, strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        ckpt_metrics = ckpt.get("metrics", {}) or {}
        best_iou = float(ckpt_metrics.get("iou", ckpt_metrics.get("val_iou", best_iou)))

    n_epochs = int((cfg.get("train", {}) or {}).get("n_epochs", 100))
    val_interval = int((cfg.get("train", {}) or {}).get("val_interval", 10))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    train_cfg = cfg.get("train", {}) or {}
    max_grad_norm = float(train_cfg.get("max_grad_norm", cfg.get("max_grad_norm", 1.0)))
    skip_nonfinite = bool(train_cfg.get("skip_nonfinite", True))
    max_consecutive_nonfinite = int(train_cfg.get("max_consecutive_nonfinite", 3))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    debug_once = bool((cfg.get("debug", {}) or {}).get("log_shapes", True))

    for epoch in range(start_epoch, n_epochs):
        model.train()
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_meters: dict[str, AverageMeter] = {"loss": AverageMeter()}
        supports_fgm_bank = _supports_fgm_bank(model)
        use_stateful_bank = supports_fgm_bank and _stateful_bank_enabled(cfg, "train")
        use_videomt_stateful = _videomt_stateful_enabled(cfg, "train")
        reset_on_new_video = _reset_bank_on_new_video(cfg)
        fgm_bank = None
        videomt_state = None
        current_video_id = None
        nonfinite_skips = 0
        consecutive_nonfinite = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            images, masks, _name = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if use_stateful_bank and isinstance(batch, dict):
                video_id = _batch_str(batch, "video_id", _name)
                should_reset = fgm_bank is None
                if reset_on_new_video:
                    should_reset = should_reset or video_id != current_video_id or _batch_bool(batch, "is_first_window", False)
                if should_reset:
                    fgm_bank = _new_fgm_bank(model)
                    current_video_id = video_id
            else:
                fgm_bank = None
            can_use_videomt_state = use_videomt_stateful and _has_videomt_state_meta(batch)
            if can_use_videomt_state:
                video_id = _batch_str(batch, "video_id", _name)
                should_reset = videomt_state is None or video_id != current_video_id or _batch_bool(batch, "is_first_window", False)
                if should_reset:
                    videomt_state = None
                    current_video_id = video_id
            else:
                videomt_state = None
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True)) and torch.cuda.is_available()):
                if _is_videomt_model(cfg):
                    out = model(
                        images,
                        mode="train",
                        epoch=epoch,
                        videomt_state=videomt_state,
                        return_videomt_state=can_use_videomt_state,
                    )
                elif supports_fgm_bank:
                    out = model(images, mode="train", fgm_bank=fgm_bank, return_fgm_bank=use_stateful_bank, epoch=epoch)
                else:
                    out = model(images, mode="train", epoch=epoch)
                if use_stateful_bank:
                    fgm_bank = out.get("fgm_bank", fgm_bank)
                if can_use_videomt_state:
                    videomt_state = out.get("videomt_state", videomt_state)
                logits, target = align_logits_masks(out["logits"], masks)
                valid_mask = _valid_mask_from_batch(batch, device)
                logits, target, aux = _filter_by_valid_mask(logits, target, out["aux"], valid_mask)
                if aux_criterion is None:
                    total_loss, loss_items = criterion(logits, target, aux=aux, epoch=epoch, include_aux=True)
                    main_loss = total_loss
                    aux_loss = total_loss * 0.0
                    main_items = loss_items
                    aux_items = {}
                    sumi_items = {}
                else:
                    main_loss, main_items = criterion(logits, target)
                    aux_loss, aux_items = aux_criterion(aux, target)
                    sumi_loss, sumi_items = sumi_criterion(aux, target, epoch=epoch, source_names=_name)
                    total_loss = main_loss + aux_loss + sumi_loss
                loss = total_loss / grad_accum

            loss_is_bad = not bool(torch.isfinite(total_loss.detach()).all().item() and torch.isfinite(loss.detach()).all().item())
            if skip_nonfinite and _distributed_any(loss_is_bad, device):
                nonfinite_skips += 1
                consecutive_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                fgm_bank = None
                videomt_state = None
                current_video_id = None
                if is_main_process():
                    print(
                        f"[warn] skipped non-finite loss at epoch {epoch:04d} step {step + 1:04d}; "
                        f"video={_batch_str(batch, 'video_id', _name) if isinstance(batch, dict) else _name} "
                        f"window={_batch_first(batch.get('window_id'), '') if isinstance(batch, dict) else ''}"
                    )
                if max_consecutive_nonfinite > 0 and consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise RuntimeError(f"stopped after {consecutive_nonfinite} consecutive non-finite losses")
                continue
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm, grad_is_finite = _grad_norm_and_finite(trainable_params, max_grad_norm)
                if skip_nonfinite and _distributed_any(not grad_is_finite, device):
                    nonfinite_skips += 1
                    consecutive_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    fgm_bank = None
                    videomt_state = None
                    current_video_id = None
                    if scaler.is_enabled():
                        scaler.update()
                    if is_main_process():
                        print(
                            f"[warn] skipped non-finite grad at epoch {epoch:04d} step {step + 1:04d}; "
                            f"grad_norm={float(grad_norm.detach().cpu())}"
                        )
                    if max_consecutive_nonfinite > 0 and consecutive_nonfinite >= max_consecutive_nonfinite:
                        raise RuntimeError(f"stopped after {consecutive_nonfinite} consecutive non-finite gradients")
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            consecutive_nonfinite = 0

            total_loss_value = float(total_loss.detach().cpu())
            batch_weight = _num_target_frames(target)
            _update_meter(train_meters, "loss", total_loss_value, batch_weight)
            for key, value in {**main_items, **aux_items, **sumi_items}.items():
                _update_meter(train_meters, key, value, batch_weight)
            metric_items = binary_metrics_from_logits(
                logits.detach().reshape(-1, 1, logits.shape[-2], logits.shape[-1]),
                target.detach().reshape(-1, 1, target.shape[-2], target.shape[-1]),
            )
            for key, value in metric_items.items():
                _update_meter(train_meters, key, value, batch_weight)
            for key, value in _debug_scalar_items(aux.get("debug", {})).items():
                _update_meter(train_meters, key, value, batch_weight)

            if debug_once and is_main_process():
                log_debug_dict("[debug shapes]", out["aux"].get("debug", {}))
                debug_once = False
            display_step = step + 1
            if is_main_process() and (display_step == 1 or display_step % int(cfg.get("log_interval", 20)) == 0):
                state_text = ""
                if use_stateful_bank and isinstance(batch, dict):
                    state_text = (
                        f" video={_batch_str(batch, 'video_id', _name)}"
                        f" window={_batch_first(batch.get('window_id'), '')}"
                        f" bank={len(fgm_bank) if fgm_bank is not None else 0}"
                    )
                elif can_use_videomt_state:
                    state_text = (
                        f" video={_batch_str(batch, 'video_id', _name)}"
                        f" window={_batch_first(batch.get('window_id'), '')}"
                        f" qstate={1 if videomt_state is not None else 0}"
                    )
                print(
                    f"[train] epoch {epoch:04d} step {display_step:04d}/{len(train_loader):04d} "
                    f"loss {train_meters['loss'].avg:.4f} "
                    f"iou {train_meters.get('iou', AverageMeter()).avg:.4f} "
                    f"f1 {train_meters.get('f1', AverageMeter()).avg:.4f}"
                    f"{state_text}"
                )
            if reset_on_new_video and use_stateful_bank and isinstance(batch, dict) and _batch_bool(batch, "is_last_window", False):
                fgm_bank = None
                current_video_id = None
            if can_use_videomt_state and _batch_bool(batch, "is_last_window", False):
                videomt_state = None
                current_video_id = None

        if scheduler is not None:
            scheduler.step()

        train_metrics = _meters_to_dict(train_meters)
        metrics = {"epoch": epoch, "train_loss": train_metrics.get("loss", 0.0)}
        metrics.update({f"train_{k}": v for k, v in train_metrics.items() if k != "loss"})
        if is_main_process():
            print(
                f"[epoch {epoch}] train loss {train_metrics.get('loss', 0.0):.4f} "
                f"f1 {train_metrics.get('f1', 0.0):.4f} iou {train_metrics.get('iou', 0.0):.4f} "
                f"nonfinite_skips {nonfinite_skips}"
            )
        if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
            if distributed:
                if is_main_process():
                    val_metrics = evaluate(_inner_model(model), main_val_loader, criterion, aux_criterion, sumi_criterion, device, cfg, epoch=epoch)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                else:
                    val_metrics = None
                _distributed_barrier(device)
            else:
                val_metrics = evaluate(model, val_loader, criterion, aux_criterion, sumi_criterion, device, cfg, epoch=epoch)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if is_main_process() and val_metrics is not None:
                print(
                    f"[epoch {epoch}] val loss {val_metrics.get('loss', 0.0):.4f} "
                    f"f1 {val_metrics.get('f1', 0.0):.4f} iou {val_metrics.get('iou', 0.0):.4f}"
                )
            if is_main_process() and val_metrics is not None and val_metrics["iou"] > best_iou:
                bad_state = _nonfinite_state_names(model)
                if bad_state:
                    print(f"[checkpoint] skipped best_iou save because model has non-finite tensors: {bad_state}")
                else:
                    best_iou = val_metrics["iou"]
                    save_checkpoint(Path(save_dir) / "best_iou.pt", model, optimizer, scheduler, epoch, val_metrics, cfg)
                    print(f"[checkpoint] best_iou updated: {best_iou:.4f}")
        else:
            val_metrics = None

        if is_main_process():
            bad_state = _nonfinite_state_names(model)
            if bad_state:
                print(f"[checkpoint] skipped latest save because model has non-finite tensors: {bad_state}")
            else:
                save_checkpoint(Path(save_dir) / "latest.pt", model, optimizer, scheduler, epoch, metrics, cfg)
            _append_epoch_log(Path(save_dir) / "log.csv", epoch, _current_lr(optimizer), train_metrics, val_metrics)
            _append_epoch_log(log_txt, epoch, _current_lr(optimizer), train_metrics, val_metrics)

    if distributed:
        _distributed_barrier(device)
