from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import VideoMTQueryMaskLoss
from src.models import B23VideoMTWindowModel
from src.utils.config import load_config


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/b23_videomt_window_final.yaml")
    model = B23VideoMTWindowModel(cfg).to(device).train()
    x = torch.rand(1, 1, 5, 3, 512, 512, device=device)
    y = torch.randint(0, 2, (1, 1, 5, 1, 512, 512), device=device).float()
    out = model(x, mode="train", return_videomt_state=True)
    assert "logits" in out
    assert "aux" in out
    assert "videomt_state" in out
    logits = out["logits"]
    aux = out["aux"]
    print("logits:", tuple(logits.shape))
    print("queries:", tuple(aux["videomt_queries"].shape))
    print("query_logits:", tuple(aux["query_logits"].shape))
    print("query_scores:", tuple(aux["query_scores"].shape))
    assert logits.shape == (1, 1, 5, 1, 512, 512)
    assert aux["videomt_queries"].shape[:4] == (1, 1, 5, 32)
    assert aux["query_logits"].shape[:4] == (1, 1, 5, 32)
    loss, items = VideoMTQueryMaskLoss(cfg["loss"])(logits, y, aux=aux, epoch=0, include_aux=True)
    assert torch.isfinite(loss)
    print("loss:", float(loss.detach().cpu()))
    print(items)
    loss.backward()
    has_query_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in model.named_parameters()
        if "query_controller" in name or "query_mask_head" in name
    )
    assert has_query_grad, "query_controller/query_mask_head has no valid gradients"
    print("query_grad: OK")


if __name__ == "__main__":
    main()
