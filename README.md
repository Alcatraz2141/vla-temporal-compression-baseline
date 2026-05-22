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

Unified episode smoke test:

```bash
uv run python scripts/smoke_test.py --sources fractal
```

After LIBERO HDF5 demos are under `data/libero_long`, inspect and smoke-test them:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
```

Compact milestone configs also work through the root wrapper:

```bash
uv run python train.py --config configs/libero_long_sliding_window.yaml
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

This includes the optional SmolVLA/LeRobot external baseline path.

## Current LIBERO Handoff

Latest completed run: corrected sliding-window LIBERO-Long continuation to 50 epochs, completed
2026-05-21.

Summary:

```text
checkpoint dir: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window
best.pt: epoch 18, val_mse 0.008474992022716574
last.pt: epoch 50, val_mse 0.015304431917944126
offline eval best.pt MSE: 0.059324943327478
offline eval best.pt MAE: 0.2940476749624525
task-5 rollout success: 0/1
task-5 rollout csv: results/libero_rollouts_sliding_window_50ep_fixed.csv
task-5 rollout video: results/rollout_videos_sliding_window_50ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Use `best.pt`, not `last.pt`, for comparisons. The run overfit after epoch 18.

Fresh RunPod restore checklist:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

uv run hf download Alcatraz1412/vla-run-backups \
  --repo-type dataset \
  --local-dir /workspace/run_backups

tar -xzf /workspace/run_backups/vla_run_artifacts_YYYYMMDD_HHMMSS.tar.gz \
  -C /root/vla-temporal-compression-baseline

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
bash libero_rollout_env/bootstrap.sh
```

Next intended commands:

```bash
# Event-gated memory
uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 --episodes-per-task 1 --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory \
  --video-every 1 --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory.csv

# Ablations
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

Before stopping a non-persistent pod:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
uv run hf upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset
```

Latest uploaded artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260521_183307.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/223967d2ad4fdacd502e714b0c2951a626909e03
```

## Superseded Event-Gated Resume Handoff

The event-gated memory run was resumed from the original 10-epoch checkpoint toward 50 epochs
and then stopped cleanly after epoch 18 before terminating a non-persistent pod.

Current state:

```text
resume config: configs/libero_long_event_gated_resume_last_to50.yaml
checkpoint dir: checkpoints/libero_long/event_gated_memory
last.pt: epoch 18, val_mse 0.012434127707300442
best.pt: epoch 6, val_mse 0.00937148479611746
main log: logs/event_gated_memory_resume_last_to50_20260521_171647.log
```

Resume on a fresh pod after restoring the latest artifact backup:

```bash
cd /root/vla-temporal-compression-baseline
uv run python train.py --config configs/libero_long_event_gated_resume_last_to50.yaml
```

After it reaches epoch 50:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_resume_last_to50.yaml \
  --checkpoint checkpoints/libero_long/event_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated_resume_last_to50.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 --episodes-per-task 1 --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory_50ep \
  --video-every 1 --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory_50ep.csv
```

## Current Event-Gated Completion Handoff

The event-gated 50-epoch continuation completed on 2026-05-22.

```text
config: configs/libero_long_event_gated_resume_last_to50.yaml
checkpoint dir: checkpoints/libero_long/event_gated_memory
log: logs/event_gated_memory_resume_last_to50_20260521_171647.log
gpu monitor: logs/event_gated_memory_resume_last_to50_gpu_monitor.log
best.pt: epoch 46, val_mse 0.008947615628130734
last.pt: epoch 50, val_mse 0.010168199252802879
artifact backup dataset: Alcatraz1412/vla-run-backups
restore rule: use the newest /workspace/run_backups/vla_run_artifacts_*.tar.gz
```

Use `best.pt` for evaluation and rollout. The A100 run used batch size `64`,
`num_workers: 8`, and `prefetch_factor: 4`.

Restore the latest artifacts on a fresh pod:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

uv run hf download Alcatraz1412/vla-run-backups \
  --repo-type dataset \
  --local-dir /workspace/run_backups

LATEST_BACKUP="$(ls -t /workspace/run_backups/vla_run_artifacts_*.tar.gz | head -1)"
tar -xzf "$LATEST_BACKUP" \
  -C /root/vla-temporal-compression-baseline

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
bash libero_rollout_env/bootstrap.sh
```

Run event-gated evaluation and the task-5 visual rollout before ablations:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_resume_last_to50.yaml \
  --checkpoint checkpoints/libero_long/event_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated_resume_last_to50.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 --episodes-per-task 1 --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory_50ep \
  --video-every 1 --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory_50ep.csv
```

Then continue the milestone ablations:

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```
