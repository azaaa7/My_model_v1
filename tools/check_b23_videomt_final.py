from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import VideoMTQueryMaskLoss
from src.models.builder import build_model
from src.train.optimizer import build_optimizer
from src.utils.config import load_config


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/b23_videomt_window_final.yaml")
    model = build_model(cfg).to(device).train()
    assert type(model).__name__ == "B23VideoMTWindowModel"
    x = torch.rand(1, 1, 5, 3, 512, 512, device=device)
    y = torch.zeros(1, 1, 5, 1, 512, 512, device=device)
    for frame_idx in range(5):
        x0 = 140 + frame_idx * 8
        y0 = 180
        y[:, :, frame_idx, :, y0 : y0 + 128, x0 : x0 + 160] = 1.0
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
    assert aux["query_scores"].shape[:4] == (1, 1, 5, 32)
    assert out["videomt_state"]["prev_q"].shape == (1, 32, 1024)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["query_logits"]).all()
    assert torch.isfinite(aux["query_scores"]).all()
    print(
        "query_scores range:",
        float(aux["query_scores"].min().detach().cpu()),
        float(aux["query_scores"].max().detach().cpu()),
    )
    loss, items = VideoMTQueryMaskLoss(cfg["loss"])(logits, y, aux=aux, epoch=0, include_aux=True)
    assert torch.isfinite(loss)
    assert items["loss_query_bce"] > 0 or items["loss_query_dice"] > 0
    assert items["loss_query_no_object"] > 0
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

    all_missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not all_missing, f"unused trainable params: {all_missing[:16]}"
    bad_grads = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert not bad_grads, f"non-finite grads: {bad_grads[:16]}"

    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and ("query_controller" in name or "query_mask_head" in name) and param.grad is None
    ]
    assert not missing, f"query final branch has unused trainable params: {missing[:8]}"
    opt = build_optimizer(model, cfg)
    group_names = {str(group.get("name", "")) for group in opt.param_groups}
    assert "query" in group_names
    assert "mask_head" in group_names
    print("optimizer groups:", sorted(group_names))
    print("[OK] final logits come from QueryMaskHead")
    print("[OK] query token enters DINOv3 final blocks")


if __name__ == "__main__":
    main()
