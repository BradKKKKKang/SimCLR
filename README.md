# SimCLR
A PyTorch implementation of SimCLR based on ICML 2020 paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).

![Network Architecture image from the paper](structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- PyYAML
```
pip install pyyaml
```
- wandb (optional)
```
pip install wandb
```

## Dataset
This repo supports `CIFAR10`, `CIFAR100`, and `STL10`. Datasets are downloaded into the `data` directory by `PyTorch` automatically unless you override `--data_root`.

For `STL10`, SimCLR pretraining uses the `train+unlabeled` split by default, while kNN and linear evaluation use the labeled `train` / `test` splits.

## Project structure
The layout now follows the same general idea as `openpi-astribot`: reusable code lives in a package, while runnable entrypoints live in `scripts/`.

```text
SimCLR/
├── assets/                  # local datasets
├── configs/                 # YAML experiment configs
├── scripts/                 # train / sweep / linear probe entrypoints
├── simclr/
│   ├── config/              # dataclass config schema + YAML loading
│   ├── data/                # dataset metadata and dataset builders
│   ├── models/              # SimCLR encoder/projector model
│   ├── training/            # losses, metrics, trainer
│   ├── evaluation/          # k-NN and linear probe evaluation
│   └── runtime/             # run directory, checkpoint, W&B, sweep
└── README.md
```

## Usage
### Train baseline / hard-positive / all-positive
The new training entrypoint is config-driven:

```bash
python scripts/train.py --config configs/cifar100_baseline.yaml
python scripts/train.py --config configs/cifar100_hard_square.yaml
python scripts/train.py --config configs/stl10_baseline.yaml
python scripts/train.py --config configs/stl10_hard_square.yaml
python -m simclr.training.trainer --config configs/cifar100_baseline.yaml
```

Warmup is controlled by `optimizer.warmup_epochs` in the YAML config, and sweep runs inherit that setting from their base config.

Each run writes:

- `config.yaml`
- `metrics.csv`
- `best_knn.pth`
- `last.pth`

into a run directory under `logging.output_dir`.

### Sweep
Local sweep supports serial execution by default and GPU-parallel execution when `CUDA_VISIBLE_DEVICES` is set:

```bash
python scripts/sweep.py --config configs/cifar100_local_sweep.yaml
python scripts/sweep.py --config configs/stl10_local_sweep.yaml
CUDA_VISIBLE_DEVICES=0,1 python scripts/sweep.py --config configs/cifar100_local_sweep.yaml
```

The sweep script reads a base config, expands the requested grid, and launches each run sequentially unless
`CUDA_VISIBLE_DEVICES` is set. When GPUs are specified, it dispatches one experiment per idle GPU and keeps polling
for newly free GPUs by checking memory usage with `nvidia-smi`.
Each expanded combination becomes one training run; it does not generate extra YAML files on disk.
Parallel sweep only launches independent single-GPU experiments. It does not use DDP, `torchrun`, or NCCL collectives.

### Linear probe
Linear probe is kept separate from the main training loop:

```bash
python scripts/linear_probe.py --model-path runs/cifar100_baseline/best_knn.pth --dataset cifar100 --data-root assets
python scripts/linear_probe.py --model-path runs/stl10_baseline/best_knn.pth --dataset stl10 --data-root assets
python -m simclr.evaluation.linear_probe --model-path runs/cifar100_baseline/best_knn.pth --dataset cifar100 --data-root assets
```

## Regularization and metrics
The minimal research framework supports three modes through `reg_weight` and `hard_ratio`:

- baseline: `reg_weight = 0.0`
- hard-positive: `0 < hard_ratio < 1.0`
- all-positive: `hard_ratio = 1.0`

The positive regularization is defined directly in code and comments:

- `linear`: `mean(1 - cos(z_i, z_j))`
- `square`: `mean((1 - cos(z_i, z_j))^2)`

The total loss is:

```text
total_loss = infonce_loss + reg_weight * reg_loss
```

Positive cosine statistics are aggregated over a whole epoch. For example,
`pos_cos_tail_mean_10` is the mean of the lowest 10% positive cosine values collected in that epoch.

For STL-10 specifically:

- pretraining uses `train+unlabeled`
- k-NN evaluation uses labeled `train` / `test`
- linear probe uses labeled `train` / `test`

## W&B logging
W&B logging is supported but configured only through environment variables. Keep `logging.use_wandb: true`
in the YAML if you want to enable it, then export:

```bash
export WANDB_PROJECT=your_project
export WANDB_ENTITY=your_entity    # optional
export WANDB_MODE=offline          # optional, defaults to offline
export WANDB_API_KEY=your_key      # optional when already logged in locally
export WANDB_DIR=/path/to/wandb    # optional
```

When `logging.use_wandb: false`, training runs normally and only writes local outputs.
Offline W&B runs can be uploaded later with:

```bash
wandb sync wandb/
```

or by syncing a specific offline run directory under `wandb/`.

## Results
There are some difference between this implementation and official implementation, the model (`ResNet50`) is trained on 
one NVIDIA TESLA V100(32G) GPU:
1. No `Gaussian blur` used;
2. `Adam` optimizer with base learning rate `5e-4` and `5` epochs of linear warmup is used to replace `LARS` optimizer;
3. No `Linear learning rate scaling` used;
4. No `CosineLR Schedule` used; after warmup, the learning rate stays constant.

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Evaluation Protocol</th>
		<th>Params (M)</th>
		<th>FLOPs (G)</th>
		<th>Feature Dim</th>
		<th>Batch Size</th>
		<th>Epoch Num</th>
		<th>τ</th>
		<th>K</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>Download</th>
		<!-- TABLE BODY -->
		<tr>
			<td align="center">KNN</td>
			<td align="center">24.62</td>
			<td align="center">1.31</td>
			<td align="center">128</td>
			<td align="center">512</td>
			<td align="center">500</td>
			<td align="center">0.5</td>
			<td align="center">200</td>
			<td align="center">89.1</td>
			<td align="center">99.6</td>
			<td align="center"><a href="https://pan.baidu.com/s/1pRwF6Uw5xnqvs2p2xQK4ZQ">model</a>&nbsp;|&nbsp;gc5k</td>
		</tr>
		<tr>
			<td align="center">Linear</td>
			<td align="center">23.52</td>
			<td align="center">1.30</td>
			<td align="center">-</td>
			<td align="center">512</td>
			<td align="center">100</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><b>92.0</b></td>
			<td align="center"><b>99.8</b></td>
			<td align="center"><a href="https://pan.baidu.com/s/1HQSNe2J-g1ptCiwKhz05cQ">model</a>&nbsp;|&nbsp;f7j2</td>
		</tr>
	</tbody>
</table>
