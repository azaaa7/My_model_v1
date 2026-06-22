from __future__ import annotations

from typing import Any

from . import B23TFCUCCMFGMLiteModel, B23TemporalRelayLiteModel, B23VideoMTWindowModel


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", "B23TFCUCCMFGMLiteModel"))
    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)
    if name == "B23TemporalRelayLiteModel":
        return B23TemporalRelayLiteModel(cfg)
    if name == "B23TFCUCCMFGMLiteModel":
        return B23TFCUCCMFGMLiteModel(cfg)
    raise ValueError(f"Unknown model name: {name}")
