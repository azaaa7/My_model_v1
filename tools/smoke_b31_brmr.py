from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.builder import build_model
from src.train.trainer import build_loss
from src.utils.config import load_config, prepare_config, resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b31_dinov3_iml_nogate_sta_brmr_lora32.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = prepare_config(load_config(resolve_config_path(args.config)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device).train()
    criterion, aux_criterion = build_loss(cfg)
    assert aux_criterion is None
    criterion = criterion.to(device)

    x = torch.randn(1, 4, 4, 3, 512, 512, device=device)
    y = torch.rand(1, 4, 4, 1, 512, 512, device=device)
    out = model(x)

    assert out["logits"].shape == (1, 4, 4, 1, 512, 512), out["logits"].shape
    assert out["aux"]["logits32"].shape == (1, 4, 4, 1, 32, 32), out["aux"]["logits32"].shape
    assert out["aux"]["logits32_coarse"].shape == (1, 4, 4, 1, 32, 32), out["aux"]["logits32_coarse"].shape
    assert out["aux"]["logits128"].shape == (1, 4, 4, 1, 128, 128), out["aux"]["logits128"].shape
    assert out["aux"]["delta32"].shape == (1, 4, 4, 1, 32, 32), out["aux"]["delta32"].shape
    assert out["aux"]["delta128"].shape == (1, 4, 4, 1, 128, 128), out["aux"]["delta128"].shape
    assert torch.isfinite(out["logits"]).all()

    loss, items = criterion(out["logits"], y, aux=out["aux"], include_aux=True)
    assert torch.isfinite(loss)
    loss.backward()

    d32 = out["aux"]["delta32"].detach().abs().mean().item()
    d128 = out["aux"]["delta128"].detach().abs().mean().item()
    assert d32 < 1.0e-4, d32
    assert d128 < 1.0e-4, d128
    print("B31 smoke OK", items, {"delta32_abs_mean": d32, "delta128_abs_mean": d128})


if __name__ == "__main__":
    main()
