from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.original_resolution import tiled_clip_logits
from src.models.builder import build_model
from src.utils.config import load_config, prepare_config, resolve_config_path


def _assert_cfg(cfg: dict[str, Any]) -> None:
    assert cfg["model"]["name"] == "B23TemporalRelayLiteModel"
    assert cfg["lora"]["rank"] == 32
    assert cfg["lora"]["alpha"] == 64
    assert cfg["lora"]["layers"] == [16, 17, 18, 19, 20, 21, 22, 23]
    local_cfg = cfg["temporal_relay"]["local_neighborhood"]
    assert local_cfg["temporal_radius"] == 1
    assert local_cfg["spatial_radius"] == 1
    assert local_cfg["relative_position_bias"] is True
    assert cfg["temporal_relay"]["global_relay"]["num_tokens"] == 2
    assert cfg["temporal_relay"]["dim"] == 256
    assert cfg["decoder"]["in_channels"] == 256
    assert cfg["temporal_relay"]["global_relay"]["num_layers"] == 1
    assert cfg["temporal_relay"]["global_relay"]["ffn_ratio"] == 1.0
    assert cfg["temporal_relay"]["recurrent_state"] is False
    assert cfg["decoder"]["mask256_head"]["enabled"] is True
    assert cfg["inference"]["original_resolution"] is True


def _lora_report(model) -> None:
    lora_names = [name for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]
    assert any("blocks.16." in name for name in lora_names)
    assert any("blocks.23." in name for name in lora_names)
    assert not any("blocks.0." in name for name in lora_names)
    assert not any("mlp.fc1" in name for name in lora_names)
    assert not any("mlp.fc2" in name for name in lora_names)
    ranks = set()
    alphas = set()
    scalings = set()
    for module in model.modules():
        if hasattr(module, "lora_A"):
            adapter = next(iter(module.lora_A.keys()))
            ranks.add(int(module.r[adapter]))
            alphas.add(int(module.lora_alpha[adapter]))
            scalings.add(float(module.scaling[adapter]))
    assert ranks == {32}, ranks
    assert alphas == {64}, alphas
    assert scalings == {2.0}, scalings
    print("LoRA: rank=32 alpha=64 scaling=2.0")


def _has_finite_grad(model, needles: tuple[str, ...]) -> bool:
    for name, p in model.named_parameters():
        if any(needle in name for needle in needles):
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all():
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b23_temporal_relay_lite.yaml")
    parser.add_argument("--skip-heavy", action="store_true", help="Only run config/import checks.")
    args = parser.parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = prepare_config(load_config(cfg_path))
    _assert_cfg(cfg)
    print(f"config OK: {Path(cfg_path).as_posix()}")
    if args.skip_heavy:
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.train()
    _lora_report(model)

    x = torch.rand(1, 1, 5, 3, 512, 512, device=device)
    out = model(x)
    assert out["logits"].shape == (1, 1, 5, 1, 512, 512), out["logits"].shape
    assert out["aux"]["boundary128"].shape == (1, 1, 5, 1, 128, 128), out["aux"]["boundary128"].shape
    debug = out["aux"]["debug"]
    assert debug["local_candidate_count"] == 27
    loss = out["logits"].mean() + out["aux"]["boundary128"].mean()
    loss.backward()

    checks = {
        "LoRA": ("lora_",),
        "local_qkv_out": (
            "temporal_relay.local_layers.0.qkv_proj",
            "temporal_relay.local_layers.0.out_proj",
        ),
        "relative_bias": ("relative_bias.relative_bias",),
        "global_relay": ("temporal_relay.relay_layers.",),
        "fusion_projection": ("temporal_relay.fusion_proj.0",),
        "feature_proj": ("feature_proj.",),
        "decoder": ("decoder.",),
    }
    for label, needles in checks.items():
        assert _has_finite_grad(model, needles), f"missing finite gradient for {label}"
        print(f"gradient OK: {label}")

    model.eval()
    for ablation in (
        {"disable_temporal_relay": True},
        {"disable_local_neighborhood": True},
        {"disable_global_relay": True},
    ):
        y = model(x, ablation=ablation)["logits"]
        assert y.shape == (1, 1, 5, 1, 512, 512)
        print(f"ablation OK: {ablation}")

    clip = torch.rand(5, 3, 544, 768, device=device)
    logits = tiled_clip_logits(model, clip, cfg)
    assert logits.shape == (5, 1, 544, 768), logits.shape
    print("original-resolution tiling OK")


if __name__ == "__main__":
    main()
