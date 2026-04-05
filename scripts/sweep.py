from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401

from simclr.config.schema import save_config
from simclr.runtime.sweep import iter_sweep_configs
from simclr.runtime.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
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
    parser = argparse.ArgumentParser(description="Run a local SimCLR parameter sweep.")
    parser.add_argument("--config", required=True, help="Path to the sweep YAML file.")
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
            f"vars={joined} sweep will launch single-process child runs with cleaned env"
        )


def query_gpu_memory_used() -> dict[str, int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for parallel GPU sweep scheduling.")

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


def find_new_run_dir(output_root: Path, run_name: str, known_dirs: set[Path]) -> Path:
    candidates = []
    if output_root.exists():
        prefix = f"{run_name}_"
        for path in output_root.iterdir():
            if not path.is_dir():
                continue
            if path in known_dirs:
                continue
            if path.name == run_name or path.name.startswith(prefix):
                candidates.append(path)
    if not candidates:
        return output_root / run_name
    return max(candidates, key=lambda path: path.stat().st_mtime)


def launch_run(
    *,
    suffix: str,
    config,
    gpu_id: str,
    temp_dir: Path,
):
    run_name = config.logging.run_name or suffix
    output_root = (REPO_ROOT / config.logging.output_dir).resolve()
    existing_dirs = set(output_root.iterdir()) if output_root.exists() else set()

    child_config = deepcopy(config)
    child_config.runtime.device = "cuda:0"
    config_path = temp_dir / f"{run_name}.yaml"
    save_config(child_config, config_path)

    process = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        cwd=REPO_ROOT,
        env=build_child_env(gpu_id),
    )
    print(f"starting_run={suffix} gpu={gpu_id}")
    return {
        "suffix": suffix,
        "gpu_id": gpu_id,
        "run_name": run_name,
        "output_root": output_root,
        "existing_dirs": existing_dirs,
        "config_path": config_path,
        "process": process,
    }


def run_parallel_sweep(args: argparse.Namespace) -> int:
    visible_gpus = parse_visible_gpus()
    if not visible_gpus:
        for suffix, config in iter_sweep_configs(args.config):
            print(f"starting_run={suffix}")
            run_dir = run_training(config)
            print(f"finished_run={run_dir}")
        return 0

    warn_if_distributed_env()
    pending_runs = list(iter_sweep_configs(args.config))
    if not pending_runs:
        return 0

    active_runs: dict[str, dict] = {}
    failed_runs: list[str] = []
    temp_dir_path = Path(tempfile.mkdtemp(prefix="simclr_sweep_configs_"))
    printed_waiting = False

    try:
        while pending_runs or active_runs:
            finished_gpus: list[str] = []
            for gpu_id, run_info in active_runs.items():
                return_code = run_info["process"].poll()
                if return_code is None:
                    continue

                run_dir = find_new_run_dir(
                    run_info["output_root"],
                    run_info["run_name"],
                    run_info["existing_dirs"],
                )
                if return_code == 0:
                    print(f"finished_run={run_dir} gpu={gpu_id} exit_code=0")
                else:
                    print(
                        f"failed_run={run_info['suffix']} gpu={gpu_id} "
                        f"exit_code={return_code} run_dir={run_dir}"
                    )
                    failed_runs.append(run_info["suffix"])
                finished_gpus.append(gpu_id)

            for gpu_id in finished_gpus:
                active_runs.pop(gpu_id, None)

            if pending_runs:
                idle_gpus = [
                    gpu_id for gpu_id in select_idle_gpus(visible_gpus, args.gpu_memory_threshold_mb)
                    if gpu_id not in active_runs
                ]
                for gpu_id in idle_gpus:
                    if not pending_runs:
                        break
                    suffix, config = pending_runs.pop(0)
                    active_runs[gpu_id] = launch_run(
                        suffix=suffix,
                        config=config,
                        gpu_id=gpu_id,
                        temp_dir=temp_dir_path,
                    )
                    printed_waiting = False

            if pending_runs and not active_runs:
                if not printed_waiting:
                    print(
                        "waiting_for_free_gpu "
                        f"pending_runs={len(pending_runs)} "
                        f"visible_gpus={','.join(visible_gpus)} "
                        f"threshold_mb={args.gpu_memory_threshold_mb}"
                    )
                    printed_waiting = True

            if pending_runs or active_runs:
                time.sleep(args.poll_interval_seconds)
    finally:
        for run_info in active_runs.values():
            if run_info["process"].poll() is None:
                run_info["process"].terminate()
        shutil.rmtree(temp_dir_path, ignore_errors=True)

    if failed_runs:
        print(f"sweep_failed_runs={','.join(failed_runs)}")
        return 1

    print("sweep_failed_runs=")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_parallel_sweep(args))

if __name__ == "__main__":
    main()
