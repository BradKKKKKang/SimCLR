from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_PROBE_SCRIPT = REPO_ROOT / "scripts" / "linear_probe.py"
DIST_ENV_VARS = [
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch linear probe over unique training experiments.")
    parser.add_argument("--dataset", required=True, help="Dataset name to filter checkpoints by.")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing training run folders.")
    parser.add_argument("--data-root", default="assets", help="Dataset root passed to linear probe.")
    parser.add_argument("--checkpoint-name", default="best_knn.pth", help="Checkpoint filename to scan for.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for linear probe results. Defaults to linear_probe_runs/<dataset>.",
    )
    parser.add_argument(
        "--gpu-memory-threshold-mb",
        type=int,
        default=1024,
        help="Treat a visible GPU as idle when its used memory is below this threshold.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="How often to poll GPU/process state while waiting for free GPUs.",
    )
    return parser.parse_args()


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} does not contain a config dictionary.")
    return payload


def parse_visible_gpus() -> list[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return []
    return [segment.strip() for segment in raw_value.split(",") if segment.strip()]


def warn_if_distributed_env() -> None:
    active_keys = [key for key in DIST_ENV_VARS if os.environ.get(key)]
    if active_keys:
        joined = ",".join(active_keys)
        print(
            "warning=distributed_env_detected "
            f"vars={joined} batch linear probe will launch single-process child runs with cleaned env"
        )


def query_gpu_memory_used() -> dict[str, int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for parallel batch linear probe scheduling.")

    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    usage_by_gpu: dict[str, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        gpu_index, memory_used = [part.strip() for part in line.split(",", maxsplit=1)]
        usage_by_gpu[gpu_index] = int(memory_used)
    return usage_by_gpu


def select_idle_gpus(visible_gpus: list[str], threshold_mb: int) -> list[str]:
    usage_by_gpu = query_gpu_memory_used()
    return [
        gpu_id
        for gpu_id in visible_gpus
        if usage_by_gpu.get(gpu_id, threshold_mb + 1) < threshold_mb
    ]


def build_child_env(gpu_id: str) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    child_env["PYTHONUNBUFFERED"] = "1"
    for key in DIST_ENV_VARS:
        child_env.pop(key, None)
    return child_env


def safe_format(value: float) -> str:
    return format(value, ".4g").replace(".", "p")


def build_effective_signature(config_dict: dict[str, Any]) -> tuple:
    dataset_name = config_dict["dataset"]["name"]
    batch_size = config_dict["runtime"]["batch_size"]
    feature_dim = config_dict["model"]["feature_dim"]
    temperature = config_dict["loss"]["temperature"]
    reg_weight = config_dict["loss"]["reg_weight"]

    if reg_weight == 0:
        return (
            dataset_name,
            "baseline",
            batch_size,
            feature_dim,
            temperature,
        )

    hard_ratio = config_dict["loss"]["hard_ratio"]
    reg_form = config_dict["loss"]["reg_form"]
    mode = "all_positive" if hard_ratio == 1.0 else "hard_positive"
    return (
        dataset_name,
        mode,
        batch_size,
        feature_dim,
        temperature,
        reg_form,
        hard_ratio,
        reg_weight,
    )


def build_experiment_tag(config_dict: dict[str, Any]) -> str:
    dataset_name = config_dict["dataset"]["name"]
    batch_size = config_dict["runtime"]["batch_size"]
    reg_weight = config_dict["loss"]["reg_weight"]

    if reg_weight == 0:
        return f"{dataset_name}_baseline_bs{batch_size}"

    reg_form = config_dict["loss"]["reg_form"]
    hard_ratio = config_dict["loss"]["hard_ratio"]
    reg_weight_str = safe_format(reg_weight)
    if hard_ratio == 1.0:
        return f"{dataset_name}_all_positive_{reg_form}_bs{batch_size}_rw{reg_weight_str}"

    hard_ratio_str = safe_format(hard_ratio)
    return (
        f"{dataset_name}_hard_positive_{reg_form}_"
        f"bs{batch_size}_hr{hard_ratio_str}_rw{reg_weight_str}"
    )


def scan_unique_checkpoints(runs_root: Path, dataset: str, checkpoint_name: str) -> list[dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}

    for checkpoint_path in sorted(runs_root.glob(f"*/{checkpoint_name}")):
        payload = load_checkpoint_payload(checkpoint_path)
        config_dict = payload.get("config")
        if not isinstance(config_dict, dict):
            print(f"skipping_checkpoint=no_config path={checkpoint_path}")
            continue
        if config_dict.get("dataset", {}).get("name") != dataset:
            continue

        signature = build_effective_signature(config_dict)
        record = {
            "checkpoint_path": checkpoint_path.resolve(),
            "config": config_dict,
            "tag": build_experiment_tag(config_dict),
            "mtime": checkpoint_path.stat().st_mtime,
        }
        existing = selected.get(signature)
        if existing is None or record["mtime"] >= existing["mtime"]:
            if existing is not None:
                print(
                    f"dedup_replaced signature={signature} "
                    f"old={existing['checkpoint_path']} new={checkpoint_path.resolve()}"
                )
            selected[signature] = record
        else:
            print(
                f"dedup_skipped signature={signature} "
                f"path={checkpoint_path.resolve()}"
            )

    records = list(selected.values())
    records.sort(key=lambda item: (item["tag"], str(item["checkpoint_path"])))
    return records


def launch_probe(
    *,
    record: dict[str, Any],
    gpu_id: str,
    args: argparse.Namespace,
    output_dir: Path,
):
    process = subprocess.Popen(
        [
            sys.executable,
            str(LINEAR_PROBE_SCRIPT),
            "--model-path",
            str(record["checkpoint_path"]),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--num-workers",
            str(args.num_workers),
            "--device",
            "cuda:0",
            "--output-dir",
            str(output_dir),
            "--run-tag",
            record["tag"],
        ],
        cwd=REPO_ROOT,
        env=build_child_env(gpu_id),
    )
    print(f"starting_linear_probe={record['tag']} gpu={gpu_id} checkpoint={record['checkpoint_path']}")
    return {
        "tag": record["tag"],
        "gpu_id": gpu_id,
        "checkpoint_path": record["checkpoint_path"],
        "process": process,
    }


def run_serial(records: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> int:
    failed: list[str] = []
    for record in records:
        print(f"starting_linear_probe={record['tag']} gpu=cpu_or_auto checkpoint={record['checkpoint_path']}")
        result = subprocess.run(
            [
                sys.executable,
                str(LINEAR_PROBE_SCRIPT),
                "--model-path",
                str(record["checkpoint_path"]),
                "--dataset",
                args.dataset,
                "--data-root",
                args.data_root,
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--num-workers",
                str(args.num_workers),
                "--device",
                "auto",
                "--output-dir",
                str(output_dir),
                "--run-tag",
                record["tag"],
            ],
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            print(f"finished_linear_probe={record['tag']} exit_code=0")
        else:
            print(f"failed_linear_probe={record['tag']} exit_code={result.returncode}")
            failed.append(record["tag"])

    if failed:
        print(f"linear_probe_failed_runs={','.join(failed)}")
        return 1
    print("linear_probe_failed_runs=")
    return 0


def run_parallel(records: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> int:
    visible_gpus = parse_visible_gpus()
    if not visible_gpus:
        return run_serial(records, args, output_dir)

    warn_if_distributed_env()
    pending = list(records)
    active: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    printed_waiting = False

    while pending or active:
        finished_gpus: list[str] = []
        for gpu_id, run_info in active.items():
            return_code = run_info["process"].poll()
            if return_code is None:
                continue
            if return_code == 0:
                print(f"finished_linear_probe={run_info['tag']} gpu={gpu_id} exit_code=0")
            else:
                print(f"failed_linear_probe={run_info['tag']} gpu={gpu_id} exit_code={return_code}")
                failed.append(run_info["tag"])
            finished_gpus.append(gpu_id)

        for gpu_id in finished_gpus:
            active.pop(gpu_id, None)

        if pending:
            idle_gpus = [
                gpu_id for gpu_id in select_idle_gpus(visible_gpus, args.gpu_memory_threshold_mb)
                if gpu_id not in active
            ]
            for gpu_id in idle_gpus:
                if not pending:
                    break
                record = pending.pop(0)
                active[gpu_id] = launch_probe(
                    record=record,
                    gpu_id=gpu_id,
                    args=args,
                    output_dir=output_dir,
                )
                printed_waiting = False

        if pending and not active and not printed_waiting:
            print(
                "waiting_for_free_gpu "
                f"pending_runs={len(pending)} "
                f"visible_gpus={','.join(visible_gpus)} "
                f"threshold_mb={args.gpu_memory_threshold_mb}"
            )
            printed_waiting = True

        if pending or active:
            time.sleep(args.poll_interval_seconds)

    if failed:
        print(f"linear_probe_failed_runs={','.join(failed)}")
        return 1
    print("linear_probe_failed_runs=")
    return 0


def main() -> None:
    args = parse_args()
    runs_root = (REPO_ROOT / args.runs_root).resolve()
    output_dir = Path(args.output_dir or (Path("linear_probe_runs") / args.dataset))
    output_dir = (REPO_ROOT / output_dir).resolve()
    records = scan_unique_checkpoints(runs_root, args.dataset, args.checkpoint_name)
    if not records:
        print(f"no_linear_probe_candidates dataset={args.dataset} runs_root={runs_root}")
        raise SystemExit(1)

    print(f"linear_probe_candidates={len(records)} dataset={args.dataset}")
    for record in records:
        print(f"linear_probe_candidate tag={record['tag']} checkpoint={record['checkpoint_path']}")

    raise SystemExit(run_parallel(records, args, output_dir))


if __name__ == "__main__":
    main()
