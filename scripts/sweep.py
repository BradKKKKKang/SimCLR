from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from simclr.runtime.sweep import iter_sweep_configs
from simclr.runtime.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local SimCLR parameter sweep.")
    parser.add_argument("--config", required=True, help="Path to the sweep YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for suffix, config in iter_sweep_configs(args.config):
        print(f"starting_run={suffix}")
        run_dir = run_training(config)
        print(f"finished_run={run_dir}")

if __name__ == "__main__":
    main()
