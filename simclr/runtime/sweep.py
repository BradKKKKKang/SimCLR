from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

import yaml

from simclr.config.schema import TrainConfig, config_to_dict, load_config, nested_update, train_config_from_dict


def load_sweep_spec(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_effective_config(config_dict: dict, base_config: TrainConfig) -> dict:
    normalized = yaml.safe_load(yaml.safe_dump(config_dict))
    if normalized["loss"]["reg_weight"] == 0:
        normalized["loss"]["hard_ratio"] = base_config.loss.hard_ratio
        normalized["loss"]["reg_form"] = base_config.loss.reg_form
    return normalized


def _build_suffix_parts(keys: list[str], config_dict: dict) -> list[str]:
    suffix_parts: list[str] = []
    skip_loss_shape = config_dict["loss"]["reg_weight"] == 0

    for key in keys:
        if skip_loss_shape and key in {"loss.hard_ratio", "loss.reg_form"}:
            continue

        cursor = config_dict
        for segment in key.split("."):
            cursor = cursor[segment]

        short_key = key.split(".")[-1]
        suffix_parts.append(f"{short_key}-{str(cursor).replace('.', 'p')}")

    return suffix_parts


def iter_sweep_configs(sweep_path: str | Path) -> Iterator[tuple[str, TrainConfig]]:
    sweep_path = Path(sweep_path)
    sweep_spec = load_sweep_spec(sweep_path)
    base_config_path = (sweep_path.parent / sweep_spec["base_config"]).resolve()
    grid = sweep_spec.get("grid", {})

    if not grid:
        raise ValueError("Sweep config must define a non-empty grid.")

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    seen_signatures: set[str] = set()

    for combo in itertools.product(*values):
        base_config = load_config(base_config_path)
        config_dict = config_to_dict(base_config)

        for key, value in zip(keys, combo):
            nested_update(config_dict, key, value)

        normalized_config_dict = _normalize_effective_config(config_dict, base_config)
        signature = yaml.safe_dump(normalized_config_dict, sort_keys=True)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        config = train_config_from_dict(normalized_config_dict)
        base_run_name = config.logging.run_name or None
        suffix_parts = _build_suffix_parts(keys, normalized_config_dict)
        suffix = "_".join(suffix_parts)
        config.logging.run_name = f"{base_run_name}_{suffix}" if base_run_name else None

        yield suffix, config
