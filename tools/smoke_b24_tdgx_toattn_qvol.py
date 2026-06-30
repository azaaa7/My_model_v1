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
    parser.add_argument("--config", default="configs/b24_dinov3_iml_tdgx_toattn_qvol_tubedrop_video_paperloss_lora32.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = prepare_config(load_config(resolve_config_path(args.config)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device).train()
    criterion, aux_criterion = build_loss(cfg)
    assert aux_criterion is None
    criterion = criterion.to(device)

    x = torch.randn(1, 4, 4, 3, 512, 512, device=device)
    y = torch.randint(0, 2, (1, 4, 4, 1, 512, 512), device=device).float()
    out = model(x)
    assert out["logits"].shape == (1, 4, 4, 1, 512, 512), out["logits"].shape
    assert out["aux"]["logits32"].shape == (1, 4, 4, 1, 32, 32), out["aux"]["logits32"].shape
    assert isinstance(out["aux"]["aux_logits"], list)
    assert isinstance(out["aux"]["aux_logits32"], list)
    assert torch.isfinite(out["logits"]).all()
    loss, items = criterion(out["logits"], y, aux=out["aux"], include_aux=True)
    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) > 0.0
    loss.backward()

    model.eval()
    with torch.no_grad():
        video = torch.randn(1, 1, 4, 3, 512, 512, device=device)
        feat = model.encode_frames(video.reshape(4, 3, 512, 512)).reshape(1, 4, model.feature_dim, 32, 32)
        model.tdgx.beta_raw.data.fill_(-20.0)
        tdgx_out, _ = model.tdgx(feat)
        assert (tdgx_out - feat).abs().max().item() < 1.0e-6
        model.temporal_only_attn.alpha_raw.data.fill_(-20.0)
        attn_out, _ = model.temporal_only_attn(feat)
        assert (attn_out - feat).abs().max().item() < 1.0e-6

    print("B24 TDGX/TOAttn/QVol smoke OK", items)


if __name__ == "__main__":
    main()
