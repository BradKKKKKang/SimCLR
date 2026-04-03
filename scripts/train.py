from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from simclr.config.schema import load_config
from simclr.runtime.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR with hard-positive regularization.")
    parser.add_argument("--config", required=True, help="Path to the YAML training config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = run_training(config)
    print(f"run_dir={run_dir}")

if __name__ == "__main__":
    main()
