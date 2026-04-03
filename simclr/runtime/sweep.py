from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

import yaml

from simclr.config.schema import TrainConfig, config_to_dict, load_config, nested_update, train_config_from_dict


def load_sweep_spec(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def iter_sweep_configs(sweep_path: str | Path) -> Iterator[tuple[str, TrainConfig]]:
    sweep_path = Path(sweep_path)
    sweep_spec = load_sweep_spec(sweep_path)
    base_config_path = (sweep_path.parent / sweep_spec["base_config"]).resolve()
    grid = sweep_spec.get("grid", {})

    if not grid:
        raise ValueError("Sweep config must define a non-empty grid.")

    keys = list(grid.keys())
    values = [grid[key] for key in keys]

    for combo in itertools.product(*values):
        base_config = load_config(base_config_path)
        config_dict = config_to_dict(base_config)
        suffix_parts = []

        for key, value in zip(keys, combo):
            nested_update(config_dict, key, value)
            short_key = key.split(".")[-1]
            suffix_parts.append(f"{short_key}-{str(value).replace('.', 'p')}")

        config = train_config_from_dict(config_dict)
        base_run_name = config.logging.run_name or None
        suffix = "_".join(suffix_parts)
        config.logging.run_name = f"{base_run_name}_{suffix}" if base_run_name else None

        yield suffix, config
