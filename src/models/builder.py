from __future__ import annotations

from typing import Any

from . import (
    B23TFCUCCMFGMLiteModel,
    B23TemporalRelayLiteModel,
    B23VideoMTWindowModel,
    B24DINOv3IMLCCMVideoModel,
    B24DINOv3IMLTDGXTOAttnQVolVideoModel,
    B25DINOv3IMLNoGateStAVideoModel,
    B31DINOv3IMLNoGateStABRMRVideoModel,
)


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", "B23TFCUCCMFGMLiteModel"))
    if name == "B24DINOv3IMLCCMVideoModel":
        return B24DINOv3IMLCCMVideoModel(cfg)
    if name == "B24DINOv3IMLTDGXTOAttnQVolVideoModel":
        return B24DINOv3IMLTDGXTOAttnQVolVideoModel(cfg)
    if name == "B25DINOv3IMLNoGateStAVideoModel":
        return B25DINOv3IMLNoGateStAVideoModel(cfg)
    if name == "B31DINOv3IMLNoGateStABRMRVideoModel":
        return B31DINOv3IMLNoGateStABRMRVideoModel(cfg)
    if name == "B23VideoMTWindowModel":
        return B23VideoMTWindowModel(cfg)
    if name == "B23TemporalRelayLiteModel":
        return B23TemporalRelayLiteModel(cfg)
    if name == "B23TFCUCCMFGMLiteModel":
        return B23TFCUCCMFGMLiteModel(cfg)
    raise ValueError(f"Unknown model name: {name}")
