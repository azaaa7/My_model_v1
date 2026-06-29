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
    parser.add_argument("--config", default="configs/b25_dinov3_iml_nogate_sta_paperloss_lora32.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = prepare_config(load_config(resolve_config_path(args.config)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device).train()
    criterion, aux_criterion = build_loss(cfg)
    assert aux_criterion is None
    criterion = criterion.to(device)

    x = torch.rand(1, 4, 4, 3, 512, 512, device=device)
    y = torch.rand(1, 4, 4, 1, 512, 512, device=device)
    out = model(x)
    assert out["logits"].shape == (1, 4, 4, 1, 512, 512), out["logits"].shape
    assert out["aux"]["logits32"].shape == (1, 4, 4, 1, 32, 32), out["aux"]["logits32"].shape
    loss, items = criterion(out["logits"], y, aux=out["aux"], include_aux=True)
    assert torch.isfinite(loss)
    loss.backward()

    groups = {"lora": 0, "adapter": 0, "decoder": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if "lora_" in name or ".lora_" in name:
            groups["lora"] += n
        elif "nogate_sta" in name:
            groups["adapter"] += n
        elif "decoder" in name:
            groups["decoder"] += n
        else:
            groups["other"] += n

    assert groups["lora"] > 0, groups
    assert groups["adapter"] > 0, groups
    assert groups["decoder"] > 0, groups
    assert "clip0_nogate_sta" in out["aux"]["debug"], out["aux"]["debug"].keys()

    print("B25 smoke OK", items)
    print(groups)


if __name__ == "__main__":
    main()
