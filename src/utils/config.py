from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg or {}
    base = cfg.pop("base", None)
    if base:
        base_path = Path(str(base)).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_cfg = load_config(base_path)
        return deep_update(base_cfg, cfg)
    return cfg


def resolve_config_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    root_p = project_root() / p
    if root_p.exists():
        return root_p.resolve()
    return p.resolve()


def resolve_path(path: str | Path | None, base_dir: Path | None = None) -> str:
    if not path:
        return "" if path is None else str(path)
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (base_dir or project_root()) / p
    return str(p.resolve())


def resolve_path_list(value: Any, base_dir: Path | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = [item.strip() for item in str(value).split(",") if item.strip()]
    return [resolve_path(item, base_dir=base_dir) for item in raw if str(item)]


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def prepare_config(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    cfg = dict(cfg)
    for key in ("train_samples", "val_samples", "test_samples"):
        if key in cfg:
            cfg[key] = resolve_path_list(cfg[key], base_dir=root)

    dinov3 = dict(cfg.get("dinov3", {}))
    if dinov3.get("repo"):
        dinov3["repo"] = resolve_path(dinov3["repo"], base_dir=root)
    if dinov3.get("weights"):
        dinov3["weights"] = resolve_path(dinov3["weights"], base_dir=root)
    cfg["dinov3"] = dinov3

    train = dict(cfg.get("train", {}))
    if train.get("save_dir"):
        train["save_dir"] = resolve_path(train["save_dir"], base_dir=root)
    cfg["train"] = train

    return cfg
