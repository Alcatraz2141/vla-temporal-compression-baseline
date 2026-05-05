# Clean PyTorch VLA Baseline

This repository is a small, modular PyTorch baseline for Franka/Panda-style Open X-Embodiment VLA research. It is intentionally centered on short temporal windows and no persistent memory, so future multi-scale memory experiments have a clean comparison point.

## Setup

```bash
uv init
uv add torch torchvision transformers datasets opencv-python tqdm pyyaml
```

The environment has already been initialized in this workspace with `uv.lock` for reproducibility.

## Layout

```text
configs/       YAML configs
data/          raw and processed episodes
datasets/      preprocessing, lazy dataset, DataLoader
models/        BC and VLA baselines
memory/        MultiScaleMemory scaffold
training/      train entrypoint
evaluation/    eval entrypoint
utils/         config, seed, metrics
scripts/       data acquisition helpers
results/       baseline metrics CSV
```

## Data

Preferred real-data route:

1. Use the official Open X-Embodiment / RT-X repository: https://github.com/google-deepmind/open_x_embodiment
2. Materialize a Franka/Panda subset from datasets such as `fractal20220817_data`, `bridge`, `taco_play`, or another Panda/Franka-compatible source.
3. Convert episodes into:

```text
data/raw/episode_000000/
  images/000000.jpg
  actions.npy
  states.npy
  metadata.json
```

Then run:

```bash
uv run python datasets/preprocess.py --input-root data/raw --output-root data/processed --max-episodes 10000 --overwrite
```

For local smoke tests without downloading RT-X:

```bash
uv run python scripts/download_data.py --synthetic
uv run python datasets/preprocess.py --input-root data/raw --output-root data/processed --overwrite
```

## WebDataset For RunPod

Export processed episodes to shard files:

```bash
uv run python scripts/export_webdataset.py \
  --input-root data/processed \
  --output-root data/webdataset \
  --T-obs 4 \
  --T-action 16 \
  --image-size 224
```

Upload the shard folder to a private Hugging Face Dataset repo:

```bash
uv run hf auth login
uv run hf repos create YOUR_HF_USERNAME/vla-franka-subset --repo-type dataset --private --exist-ok
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf upload-large-folder \
  YOUR_HF_USERNAME/vla-franka-subset \
  data/webdataset \
  --repo-type dataset
```

The same flow is also available through the helper script:

```bash
uv run python scripts/hf_dataset.py create YOUR_HF_USERNAME/vla-franka-subset
uv run python scripts/hf_dataset.py upload YOUR_HF_USERNAME/vla-franka-subset --local-dir data/webdataset
```

On RunPod, download the processed shards:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download \
  YOUR_HF_USERNAME/vla-franka-subset \
  --repo-type dataset \
  --local-dir data/webdataset
```

or:

```bash
uv run python scripts/hf_dataset.py download YOUR_HF_USERNAME/vla-franka-subset --local-dir data/webdataset
```

Then train from WebDataset shards:

```bash
uv run python training/train.py --config configs/webdataset_smoke.yaml
```

## Train And Evaluate

```bash
uv run python training/train.py --config configs/default.yaml --baseline sliding_window
uv run python evaluation/eval.py --config configs/default.yaml
```

Baseline variants:

```bash
uv run python training/train.py --config configs/default.yaml --baseline no_temporal
uv run python training/train.py --config configs/default.yaml --baseline larger_window
uv run python training/train.py --config configs/default.yaml --baseline bc_resnet50
```

Smoke test:

```bash
uv run python training/train.py --config configs/smoke.yaml
uv run python evaluation/eval.py --config configs/smoke.yaml
```

Evaluation appends fixed-seed metrics to `results/baselines.csv`. Success rate is recorded as `NaN` unless connected to an actual rollout environment.

## Memory Hook

`memory/multiscale_memory.py` exposes:

```python
class MultiScaleMemory(nn.Module):
    def forward(self, current_tokens, past_memory):
        return updated_tokens, updated_memory
```

The baseline model can enable it with:

```yaml
model:
  use_memory: true
```

The hook is deliberately scaffolded only. It marks the insertion point for short-term buffers, mid-term summaries, long-term slots, KV-cache experiments, and summary tokens.

## Real Experiments

For larger RunPod runs, WebDataset sweeps, and simulator rollout wiring, see:

```text
docs/runpod_experiments.md
```
