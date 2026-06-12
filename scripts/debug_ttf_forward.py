from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import B23TFCUCCMFGMLiteModel
from src.train.trainer import build_loss
from src.utils.config import load_config, prepare_config, resolve_config_path
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b23_ttf_ccm_structloss_lora32.yml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--loss-size", type=int, default=128)
    parser.add_argument("--num-clips", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_loss_checks(cfg: dict, device: torch.device, size: int) -> None:
    criterion, aux_criterion = build_loss(cfg)
    if aux_criterion is not None:
        raise RuntimeError("debug_ttf_forward expects a unified loss such as composite_forensic or ttf_minimal")
    criterion = criterion.to(device)
    b, m, k = 1, int(cfg.get("num_clips", 4)), int(cfg.get("num_frames", 4))
    logits = torch.randn(b, m, k, 1, size, size, device=device)
    aux = {
        "mask128": torch.randn(b, m, k, 1, 128, 128, device=device),
        "mask256": torch.randn(b, m, k, 1, 256, 256, device=device),
        "ccm_mask32": torch.randn(b, m, k, 1, 32, 32, device=device),
        "boundary128": torch.randn(b, m, k, 1, 128, 128, device=device),
        "ttf_residual_energy": torch.zeros(b, m, k, 32 * 32, device=device),
    }
    cases = {
        "random": torch.randint(0, 2, (b, m, k, 1, size, size), device=device).float(),
        "empty": torch.zeros(b, m, k, 1, size, size, device=device),
        "full": torch.ones(b, m, k, 1, size, size, device=device),
    }
    for name, target in cases.items():
        loss, items = criterion(logits, target, aux=aux, epoch=20, include_aux=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"CompositeForensicLoss is non-finite for case={name}: {loss}")
        print(f"[loss:{name}] {float(loss.detach().cpu()):.6f} keys={sorted(items)}")


@torch.no_grad()
def run_model_check(cfg: dict, device: torch.device, num_clips: int | None, num_frames: int | None) -> None:
    cfg = prepare_config(cfg)
    model = B23TFCUCCMFGMLiteModel(cfg).to(device)
    model.eval()
    b = 1
    m = int(num_clips or cfg.get("num_clips", 4))
    k = int(num_frames or cfg.get("num_frames", 4))
    size = int(cfg.get("input_size", 512))
    video = torch.rand(b, m, k, 3, size, size, device=device)
    out = model(video, mode="debug")
    expected = (b, m, k, 1, size, size)
    actual = tuple(out["logits"].shape)
    if actual != expected:
        raise RuntimeError(f"Expected logits shape {expected}, got {actual}")
    temporal_debug = (out.get("aux", {}).get("debug", {}) or {}).get("temporal_encoder", {})
    print(f"[model] logits={actual}")
    print(f"[model] temporal_debug={temporal_debug}")


def main():
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 666666)))
    device = select_device(args.device)
    print(f"[debug] config={cfg_path}")
    print(f"[debug] device={device}")
    run_loss_checks(cfg, device, args.loss_size)
    if not args.skip_model:
        run_model_check(cfg, device, args.num_clips, args.num_frames)


if __name__ == "__main__":
    main()
