# RunPod Experiment Path

This repo now has two experiment paths:

1. Offline action prediction on Open X / RT-X validation shards.
2. Simulator rollouts through `evaluation/rollout.py` for Gymnasium-style envs.

## Recommended Large-Data Flow

Materialize a real subset:

```bash
uv sync --extra openx
uv run python scripts/openx_rlds_to_raw.py \
  --dataset fractal20220817_data \
  --max-episodes 5000 \
  --output-root data/raw \
  --overwrite
```

Preprocess and export shards:

```bash
uv run python datasets/preprocess.py \
  --input-root data/raw \
  --output-root data/processed \
  --max-episodes 5000 \
  --overwrite

uv run python scripts/export_webdataset.py \
  --input-root data/processed \
  --output-root data/webdataset \
  --T-obs 4 \
  --T-action 16 \
  --image-size 224 \
  --max-samples-per-shard 2048
```

Upload processed shards once:

```bash
uv run hf auth login
uv run python scripts/hf_dataset.py create YOUR_HF_USERNAME/vla-franka-subset
uv run python scripts/hf_dataset.py upload YOUR_HF_USERNAME/vla-franka-subset --local-dir data/webdataset
```

On RunPod, download only processed shards:

```bash
uv sync
uv run hf auth login
uv run python scripts/hf_dataset.py download YOUR_HF_USERNAME/vla-franka-subset --local-dir data/webdataset
```

## Baseline Sweep

Edit `configs/runpod_webdataset.yaml` so `train_urls` / `val_urls` match your shard count. Then run:

```bash
uv run python scripts/run_baselines.py --sweep configs/baseline_sweep.yaml
```

This trains and evaluates:

- `no_temporal`
- `sliding_window`
- `larger_window`
- `bc_resnet50`
- `rt1_style`

`rt1_style` is an RT-1-inspired controlled PyTorch baseline using CNN visual features, proprioceptive tokens,
temporal embeddings, causal Transformer layers, and chunked continuous action prediction. It is intended for
fair comparison inside this repo, not as an exact reproduction of Google's full RT-1 recipe.

## SmolVLA External Baseline

SmolVLA is handled as an external LeRobot baseline because it uses the LeRobot dataset/training stack rather
than this repo's PyTorch `training/train.py` loop. This preserves the existing baselines while adding the
pretrained open-source VLA baseline requested in the roadmap.

Install SmolVLA in a separate environment. Do not install LeRobot into the main baseline `.venv`: current
LeRobot releases pin older Torch / Hugging Face dependency ranges than this repo's PyTorch baseline stack.

```bash
uv venv .venv-smolvla --python 3.11
source .venv-smolvla/bin/activate
uv pip install -r requirements/smolvla.txt
```

Export processed episodes to LeRobot format:

```bash
python scripts/export_lerobot.py \
  --config configs/smolvla.yaml \
  --train-repo-id YOUR_HF_USERNAME/vla-franka-lerobot-train \
  --val-repo-id YOUR_HF_USERNAME/vla-franka-lerobot-val \
  --overwrite
```

Push during export by adding `--push`, or upload/push with LeRobot tools after inspecting the local dataset.

Run zero-shot SmolVLA offline action prediction on the validation LeRobot dataset:

```bash
python scripts/smolvla_baseline.py eval --config configs/smolvla.yaml
```

Fine-tune SmolVLA:

```bash
python scripts/smolvla_baseline.py train --config configs/smolvla.yaml
```

Evaluate a fine-tuned checkpoint:

```bash
python scripts/smolvla_baseline.py eval \
  --config configs/smolvla.yaml \
  --policy-path outputs/train/smolvla_openx/checkpoints/last/pretrained_model
```

Before running, set these fields in `configs/smolvla.yaml`:

```yaml
lerobot_export:
  train_repo_id: YOUR_HF_USERNAME/vla-franka-lerobot-train
  val_repo_id: YOUR_HF_USERNAME/vla-franka-lerobot-val
smolvla:
  train_dataset_repo_id: YOUR_HF_USERNAME/vla-franka-lerobot-train
  val_dataset_repo_id: YOUR_HF_USERNAME/vla-franka-lerobot-val
```

The SmolVLA offline metric is MSE/MAE against validation actions. It is useful as a roadmap-aligned pretrained
VLA baseline, but true task Success Rate still requires a robot or simulator rollout benchmark.

Metrics append to `results/baselines.csv`.

## GPU Utilization Knobs

Start with:

```yaml
data:
  num_workers: 8
  prefetch_factor: 4
training:
  batch_size: 64
  amp: true
  amp_dtype: bfloat16
  channels_last: true
```

If GPU utilization is still low:

- Increase `batch_size` until VRAM is near 80-90%.
- Increase `data.num_workers` to 12 or 16 if CPU has enough cores.
- Store shards on the pod's local volume, not network-mounted slow disk.
- Use more/larger shards; tiny datasets finish CPU decode too quickly to saturate a GPU.
- Leave `compile: false` for first runs; try `compile: true` after shapes/config are stable.

## Simulator Rollouts

Install the simulator separately, then use a Gymnasium-compatible env id:

```bash
uv run python evaluation/rollout.py \
  --config configs/sim_rollout.yaml \
  --env-id YOUR_ENV_ID \
  --checkpoint checkpoints/runpod_webdataset/sliding_window/best.pt \
  --episodes 20
```

`evaluation/rollout.py` expects the env observation to provide an RGB image and robot state. Configure:

```yaml
rollout:
  image_key: image
  state_key: state
  success_key: success
```

If the env uses different names, change those keys. Success is read from `info[success_key]`, or from a reward threshold if you set `reward_success_threshold`.
