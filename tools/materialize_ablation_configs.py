from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = config_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f) or {}

    runtime = cfg_all.get("runtime", cfg_all)
    presets = cfg_all.get("ablation_presets", {}) or {}
    if not presets:
        raise ValueError(f"No ablation_presets found in {config_path}")

    for preset_name, preset in presets.items():
        cfg = copy.deepcopy(runtime)
        deep_update(cfg, copy.deepcopy((preset or {}).get("overrides", {}) or {}))
        out_path = out_dir / f"{preset_name}.yml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[materialize] {preset_name} -> {out_path}")


if __name__ == "__main__":
    main()
