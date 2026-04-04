from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from simclr.data.datasets import DATASET_REGISTRY
from simclr.evaluation.linear_probe import run_linear_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe evaluation for SimCLR checkpoints.")
    parser.add_argument("--model-path", required=True, help="Path to best_knn.pth or last.pth.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--data-root", default="assets")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="linear_probe_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, best_model_path = run_linear_probe(
        model_path=args.model_path,
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"linear_probe_csv={output_path}")
    print(f"linear_probe_best_model={best_model_path}")

if __name__ == "__main__":
    main()
