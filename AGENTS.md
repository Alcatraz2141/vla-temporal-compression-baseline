# AGENTS.md

This file is the operating guide for autoresearch agents working on this repository. The project is serious research code, not a toy demo. Preserve existing pipelines unless a change is explicitly needed for the current milestone.

## Research Goal

We are building a publishable long-horizon VLA / behavior-cloning benchmark for robot manipulation.

The core hypothesis is:

> Embodied trajectories have non-uniform temporal relevance. Recent interaction should remain high fidelity, while older context can be compressed into lightweight event/state summaries. A hierarchical memory policy should outperform strict short-context sliding-window policies on long-horizon tasks where the current observation alone is insufficient.

The baseline limitation we are targeting is that many current VLA-style policies operate over short temporal windows and do not maintain persistent memory. This repository should let us compare that standard setup against multi-scale memory in a controlled, reproducible way.

The first publishable milestone is:

1. Run a strict `sliding_window` baseline on LIBERO-Long.
2. Run an `event_gated_memory` model on the same train/val/test split.
3. Run small, focused ablations:
   - age-based gate instead of event gate
   - concat query instead of cross-attention query
4. Log offline action-prediction metrics consistently.
5. Later add simulator rollout success-rate evaluation once LIBERO sim is wired cleanly.

Do not jump to every possible ablation before the first table is stable.

## Current Project State

The repository contains a PyTorch behavior-cloning pipeline with:

- config-driven training
- config-driven evaluation
- local episode datasets
- WebDataset support from earlier experiments
- Hugging Face upload/download workflow
- synthetic smoke tests from the original setup
- Open X / RT-X-style offline data support
- LIBERO-Long HDF5 reading through direct `h5py`
- a unified episode-level loader for current milestone experiments

Latest paper-seed task-2 phase-ACT state as of 2026-06-23:

```text
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
protocol: train10 / val5 / test5 split-aware LIBERO rollouts, max steps 300
seed 43 phase ACT: best epoch 57, offline continuous_mse 0.01966886719018221, rollout 17/20
seed 44 phase ACT: best epoch 58, offline continuous_mse 0.020462753280997278, rollout 10/20
summary: results/task2_phase_act_paper_seeds_20260623.md
```

This is seed-variance evidence for phase ACT. Do not claim that longer phase-ACT training
reliably explains the task-2 memory result until matched event-gated seed-43/44 runs are complete.

Additional random phase-ACT seed completed on 2026-06-24:

```text
seed 187: best epoch 58, offline continuous_mse 0.019253734470903873
unique rollout: train30 20/30, val5 1/5, test5 4/5 = 25/40
held-out val+test: 5/10
```

The task-2 phase seeds have similar offline metrics but materially different closed-loop success.
For the paper comparison, train event-gated ACT from scratch for 60 epochs with matched seeds
43, 44, and 187. Use the same unique train30/val5/test5 rollout protocol before age-gated controls.

Latest artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260624_121312.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/bf0eecd96297788fabe223fbb6099b1735703945
```

Current from-scratch event-gated paper run as of 2026-06-24:

```text
seed: 43
config: configs/paper_event_gated_act_task2_seed43.yaml
resume config: configs/paper_event_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory/last.pt
stopped safely after epoch: 21
epoch 21 val_mse: 0.04350930461883545
best epoch: 21
best_val: 0.04350930461883545
```

Resume with the resume config; the unfinished epoch 22 work was intentionally discarded.

Epoch-21 shutdown backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260624_185606.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/02e537abdbdf6587d8f268517b35eb36290a32a3
```

Task-2 from-scratch event-gated seed-43 completion as of 2026-06-26:

```text
config: configs/paper_event_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory/best.pt
completed epoch: 60
best epoch: 58
best val_mse: 0.020923468711972235
offline continuous_mse: 0.01848969299942255
offline continuous_mae: 0.09646275240182876
gripper_sign_accuracy: 0.9892899991989136
rollout train30 / val5 / test5: 18/30, 3/5, 3/5 = 24/40
held-out val+test: 6/10
summary: results/paper_event_gated_task2_seed43_20260626.md
artifact backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/db21d6240870b8a95ecdc8b39337f8a2691faad4
```

This from-scratch event-gated seed improves offline continuous MSE versus seed-43 phase ACT, but
does not reproduce seed-43 phase ACT's stronger online result. Comparable 20-episode subset:

```text
seed-43 phase ACT:                train10 8/10, val5 4/5, test5 5/5 = 17/20
seed-43 event-gated from scratch: train10 6/10, val5 3/5, test5 3/5 = 12/20
```

Do not claim seed-43 from-scratch event memory improves the phase baseline. Continue the matched
from-scratch protocol with seeds 44 and 187 before age-gated controls.

Task-2 from-scratch event-gated seed-44 partial run as of 2026-06-27:

```text
config: configs/paper_event_gated_act_task2_seed44.yaml
resume config: configs/paper_event_gated_act_task2_seed44_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
stopped by request during epoch: 12
last completed epoch: 11
best epoch: 11
best/last val_mse: 0.06793733193278313
log: logs/paper_event_gated_task2_seed44_20260627_0606.log
```

Resume seed 44 with the resume config. The partial epoch-12 work was intentionally discarded.

Task-2 from-scratch event-gated seed-44 restarted and evaluated as of 2026-06-28:

```text
config: configs/paper_event_gated_act_task2_seed44.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
completed/stopped epoch: 50
best checkpoint by decoupled quick validation: epoch 46
epoch-50 offline continuous_mse: 0.04505929201841354
epoch-50 offline continuous_mae: 0.14305126070976257
epoch-50 gripper_sign_accuracy: 0.989312493801117
epoch-50 rollout train30 / val5 / test5: 17/30, 3/5, 4/5 = 24/40
epoch-50 held-out val+test: 7/10
epoch-46 best.pt rollout train30 / val5 / test5: 17/30, 2/5, 2/5 = 21/40
summary: results/paper_event_gated_task2_seed44_epoch50_20260628.md
```

Use epoch-50 `last.pt` for seed-44 event-gated reporting unless a later audit supersedes it.
The decoupled quick-validation `best.pt` did not select the stronger rollout checkpoint.

Task-2 from-scratch event-gated seed-44 was continued to epoch 60 and evaluated on 2026-06-29:

```text
config: configs/paper_event_gated_act_task2_seed44_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
completed epoch: 60
epoch-60 offline continuous_mse: 0.04118259623646736
epoch-60 offline continuous_mae: 0.13802042752504348
epoch-60 gripper_sign_accuracy: 0.9870000004768371
epoch-60 rollout train30 / val5 / test5: 16/30, 1/5, 1/5 = 18/40
epoch-60 held-out val+test: 2/10
summary: results/paper_event_gated_task2_seed44_epoch60_20260629.md
```

Epoch 60 improved offline action prediction but worsened rollout substantially. Keep epoch-50
`last.pt` as the seed-44 event-gated reporting checkpoint unless a later rollout audit supersedes it.

Important speed/protocol update from 2026-06-28:

```text
validation decoupling: build_dataloader now uses shuffle/train intent, not split name, to decide
training mode. This fixes val_split=train accidentally running a full 20k stochastic train-mode
validation pass with augmentation.

seed-44 event-gated epoch timing after fix:
  train: about 10.2 minutes/epoch
  validation: about 3-4 seconds/epoch

previous behavior:
  validation added roughly another 9-10 minutes/epoch.
```

This change affects training-time validation and `best.pt` selection, not model forward logic or
rollout. For rollout-facing comparisons, evaluate both promising checkpoints when validation
protocol changes.

Frozen-vision speed diagnostic on seed 44 was also run and stopped on 2026-06-27:

```text
config: configs/diagnostic_event_gated_act_task2_seed44_freeze_vision.yaml
checkpoint: checkpoints/diagnostic_event_gated_task2_seed44_freeze_vision/diagnostic_event_gated_act_h20_task2_freeze_vision_seed44/best.pt
stopped by request during epoch: 8
last completed epoch: 7
best/last val_mse: 0.14847677819728852
log: logs/diagnostic_event_gated_task2_seed44_freeze_vision_20260627.log
gpu monitor: results/diagnostic_event_gated_task2_seed44_freeze_vision_gpu.csv
```

Finding: freezing ResNet18 reduced VRAM substantially but did not materially improve wall-clock
epoch time. The frozen run stayed around 16.5-17 minutes per epoch, similar to the unfrozen seed-44
run, and learned more slowly. GPU utilization remained bursty with many 0% samples, so the current
bottleneck is the older-context image/data/validation pipeline rather than ResNet backward compute.
Do not use full frozen-vision as the default paper protocol. Next speed diagnostic should test
larger task-2 episode cache / validation timing before changing model or memory settings.

Artifact backup after the seed-44 stop and frozen-vision diagnostic:

```text
local: /workspace/run_backups/vla_run_artifacts_20260627_113647.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/513bfd80a278f3f22d0b874f70709bf36aecc147
```

Important: do not remove or break older baselines or the old WebDataset path. The current milestone adds a better episode-level path; it does not delete the previous work.

## Main Data Philosophy

Earlier WebDataset exports stored overlapping window samples. That caused repeated image storage and made long-horizon memory harder to study cleanly.

The new milestone uses an episode-level format:

```text
episode_id
frames / frame paths / HDF5 frame dataset
actions [T, action_dim]
states / proprio [T, state_dim]
language / task metadata
timestamps if available
```

Windows are constructed inside the dataloader, not pre-exported as overlapping samples. This is required to avoid split leakage and to support memory experiments.

Each training sample should have this model-agnostic structure:

```python
{
    "recent_obs":      # (K_recent, C, H, W)
    "recent_actions":  # (K_recent, action_dim)
    "recent_states":   # (K_recent, state_dim)
    "older_obs":       # (T_older, C, H, W)
    "older_actions":   # (T_older, action_dim)
    "older_states":    # (T_older, state_dim)
    "target_actions":  # (H_action, action_dim)
    "language":        # string
    "episode_id":      # string
    "timestep":        # int
    "recent_mask":     # optional mask
    "older_mask":      # optional mask
    "target_mask":     # optional mask
}
```

## Current Data Sources

### LIBERO-Long

LIBERO-Long is the primary benchmark for the memory milestone.

Current inspected real LIBERO HDF5 structure:

```text
root keys: ['data']
episode key: data/demo_0
obs keys:
  agentview_rgb
  ee_ori
  ee_pos
  ee_states
  eye_in_hand_rgb
  gripper_states
  joint_states
image shape: (272, 128, 128, 3), dtype=uint8
action shape: (272, 7), dtype=float64
selected proprio/state dim: 8
action dim: 7
```

The state/proprio vector is built from:

```text
ee_pos + ee_ori + gripper_states
```

The LIBERO import itself is optional. The repository currently uses direct `h5py` access because this is simpler and more robust for offline BC experiments. If official LIBERO simulation is added later, wrap the import in a clear optional dependency check.

Do not upload LIBERO data into the user Hugging Face repository unless explicitly requested. Prefer downloading directly from the official Hugging Face dataset repo:

```text
yifengzhu-hf/LIBERO-datasets
```

### Open X / RT-X Offline Episodes

There are approximately 6000 offline Open X-style episodes available or being built:

- about 5000 Franka / Panda episodes
- about 900 diverse-arm episodes currently downloaded, mostly from UR5-style data
- about 4000 additional diverse-arm episodes intended to be downloaded later

These Open X episodes are useful for:

- offline behavior cloning
- pretraining / warmup
- action-prediction baselines
- testing whether memory improves supervised prediction on longer trajectories

They are not directly usable as LIBERO-Long success rollouts. They use different robot embodiments, cameras, action spaces, tasks, and success definitions. A model trained on Open X does not automatically become meaningful in LIBERO sim without LIBERO-format training/fine-tuning or careful adaptation.

Use Open X as offline pretraining data. Use LIBERO for the primary benchmark and eventual success-rate rollouts.

## Implemented Files

### `datasets/episode_loader.py`

This is the unified milestone dataset loader.

It supports:

- `source="libero_long"`
- `source="fractal"`
- `source="ur5"`
- `source="others"`

Expected source locations:

```text
libero_long -> data/libero_long/**/*.hdf5
fractal     -> data/raw
ur5         -> data/raw_diverse
others      -> data/raw_diverse
```

Responsibilities:

- deterministic episode-level 80/10/10 splits
- writes split files under `splits/{source}_{train,val,test}.txt`
- no split leakage across overlapping windows
- skips episodes shorter than `K_recent + H_action`
- constructs recent context, older context, and target action chunk inside `__getitem__`
- normalizes images to `[0, 1]`
- resizes images to `image_size`, usually `128`
- returns consistent batch fields across all sources

### `datasets/data_loader.py`

Wires the existing training stack to the unified episode loader.

Use this route through configs:

```yaml
data:
  source: unified_episode
  dataset: libero_long
```

### `scripts/inspect_libero.py`

Inspects one LIBERO HDF5 file and prints:

- root keys
- episode key
- observation keys
- image shape
- action shape
- state/proprio component shapes
- state dim
- action dim
- episode length
- language/task label

Use this before training whenever LIBERO data changes.

### `scripts/smoke_test.py`

Loads five windows per requested source, prints batch shapes, checks tensors for NaNs, and writes results to:

```text
results/smoke_test.csv
```

Known successful LIBERO smoke-test shape:

```text
recent_obs:      [5, 8, 3, 128, 128]
recent_actions:  [5, 8, 7]
recent_states:   [5, 8, 8]
older_obs:       [5, 64, 3, 128, 128]
older_actions:   [5, 64, 7]
older_states:    [5, 64, 8]
target_actions:  [5, 4, 7]
timestep:        [5]
recent_mask:     [5, 8]
older_mask:      [5, 64]
target_mask:     [5, 4]
```

### `train.py`

Root wrapper for:

```text
training/train.py
```

This allows:

```bash
uv run python train.py --config configs/libero_long_sliding_window.yaml
```

### `configs/libero_long_sliding_window.yaml`

Primary short-context baseline config for LIBERO-Long.

Important settings:

```yaml
dataset: libero_long
baseline: sliding_window
K_recent: 8
H_action: 4
chunk_size: 4
max_memory_tokens: 16
gate_type: none
query_type: none
seed: 42
image_size: 128
```

The model should ignore `older_*` fields.

### `configs/libero_long_event_gated.yaml`

Primary proposed memory model config.

Important settings:

```yaml
dataset: libero_long
baseline: event_gated_memory
K_recent: 8
H_action: 4
chunk_size: 4
max_memory_tokens: 16
gate_type: event
query_type: cross_attention
seed: 42
image_size: 128
```

### `configs/ablation_gate_age.yaml`

Same as event-gated memory, but the gate is age-based:

```yaml
gate_type: age_based
query_type: cross_attention
```

Purpose: prove that event-aware selection matters, not just recency.

### `configs/ablation_query_concat.yaml`

Same as event-gated memory, but memory is queried by concatenation:

```yaml
gate_type: event
query_type: concat
```

Purpose: test whether cross-attention query is better than simply appending memory tokens.

### `configs/pretrain_fractal.yaml`

Config-only for later offline pretraining on Franka / RT-1-style data.

Do not treat this as a finished publishable run until the source data and evaluation protocol are stable.

## Baselines and Model Approaches

### Sliding Window Baseline

This is the strict baseline for the first paper table.

It receives only the last `K_recent` observations/states/actions and predicts the next `H_action` actions. It ignores older trajectory context.

This represents the short-context limitation in many VLA-style systems.

Expected use:

```text
LIBERO-Long action prediction baseline
Open X offline action prediction baseline
```

### Event-Gated Hierarchical Memory

This is the proposed method for the first milestone.

High-level behavior:

1. Encode each timestep into a step token from visual, action, and state features.
2. Keep recent `K_recent` steps as full-fidelity tokens.
3. Divide older context into temporal chunks.
4. Compress each older chunk into summary token(s) using lightweight attention pooling.
5. Score memory chunks using event deltas and age.
6. Keep/weight the most relevant memory tokens up to `max_memory_tokens`.
7. Query memory with cross-attention or concat transformer.
8. Predict continuous action chunks.

The event gate should use cheap signals already available from embeddings:

```text
delta_visual
delta_action
delta_state
age
```

Do not add a large VLM, generative summarizer, or expensive world model inside the memory update.

### Age-Gated Ablation

This uses a memory gate based primarily on recency/age.

Purpose:

```text
If age-gated performs similarly to event-gated, the proposed event mechanism may not be doing enough.
If event-gated performs better, it supports the claim that task-relevant events matter beyond simple recency.
```

### Query Concatenation Ablation

This replaces cross-attention memory query with token concatenation.

Purpose:

```text
Test whether the model benefits from an explicit current-state-to-memory query mechanism.
```

### Existing Baselines Not To Break

These existed before the LIBERO milestone and should remain runnable:

- `bc_resnet50`
- `no_temporal`
- `sliding_window`
- `rt1_style`

Do not refactor these unless required by a failing test or explicit user request.

### Later Models Not Implemented Yet

Do not implement these until the first milestone table is stable:

- RB-VLA-style recursive belief baseline
- SmolVLA integration
- full OpenVLA/Octo inference integration
- compression-ratio sweeps
- memory-content sweeps
- full LIBERO simulator rollouts

## Metrics

Current offline metrics:

- MSE
- MAE
- validation loss
- predicted temporal smoothness
- per-episode metrics where available

Success rate is not yet a real metric unless LIBERO simulator rollouts are wired. Until then, report success rate as unavailable / not evaluated, not zero.

For publication-quality claims, eventually add:

- LIBERO rollout success rate
- task-level success breakdown
- multiple seeds
- confidence intervals or standard deviations

## Acceptance Criteria For Current Milestone

Before claiming the milestone is complete:

1. LIBERO HDF5 inspection succeeds.
2. LIBERO smoke test succeeds on at least five windows.
3. Sliding-window train run starts and loss decreases.
4. Event-gated train run starts and loss decreases.
5. Evaluation writes rows to `results/baselines.csv`.
6. Gate and query ablation configs run without code changes.
7. All runs use fixed seed `42`.
8. No split leakage exists because split files are episode-level.
9. Checkpoints are stored in separate directories per run.
10. The README or this file contains the exact commands used.

## Agent Rules

- Keep the pipeline reproducible with `uv`.
- Prefer local CPU smoke tests before RunPod jobs.
- Never mix train/val/test windows from the same episode.
- Do not silently change action/state dimensions.
- Do not delete old WebDataset code.
- Do not upload huge datasets unless the user explicitly asks.
- Do not re-upload LIBERO to the user's Hugging Face repo; download it from the official repo on each machine.
- Do not use Open X offline episodes as LIBERO rollout results.
- Keep configs explicit and nested. Do not collapse them into minimal flat YAMLs unless the user asks.
- If a run fails, inspect data shapes first, then model input assumptions, then config normalization.
- If CUDA utilization is low, check dataloader workers, image decode path, batch size, and disk/network bandwidth before changing the model.

## Local Setup Commands

From the repo root:

```bash
cd /Users/ruchirtidke/Work/Research/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

On RunPod:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Download LIBERO-Long

Download the full LIBERO-Long / `libero_10` HDF5 suite:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2
```

If bandwidth/storage is limited, download one file first:

```bash
uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5"
```

Inspect the downloaded data:

```bash
uv run python scripts/inspect_libero.py --data-root data/libero_long
```

Expected output should include:

```text
image shape: (..., 128, 128, 3)
action shape: (..., 7)
state/proprio dim: 8
action dim: 7
```

## Smoke Tests

Run LIBERO smoke test:

```bash
uv run python scripts/smoke_test.py --sources libero_long
```

Run Open X / offline-data smoke tests:

```bash
uv run python scripts/smoke_test.py --sources fractal ur5 others
```

Run all available sources:

```bash
uv run python scripts/smoke_test.py --sources libero_long fractal ur5 others
```

Results are written to:

```text
results/smoke_test.csv
```

## Train And Evaluate LIBERO Baselines

### Sliding Window

```bash
uv run python train.py --config configs/libero_long_sliding_window.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window.yaml
```

### Event-Gated Memory

```bash
uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml
```

### Age-Gated Ablation

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml
```

### Query-Concat Ablation

```bash
uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

Expected result path:

```text
results/baselines.csv
```

## Offline Open X Training

Current offline data status:

```text
~5000 Franka/Panda episodes
~900 diverse-arm episodes
~4000 more diverse-arm episodes intended to be downloaded
~6000 total current Open X-style offline episodes
```

These are not directly compatible with LIBERO rollout success. Use them for offline pretraining or supervised action prediction.

Smoke test:

```bash
uv run python scripts/smoke_test.py --sources fractal ur5 others
```

Pretraining config-only run:

```bash
uv run python train.py --config configs/pretrain_fractal.yaml
```

Do not claim this as a LIBERO result. It is Open X offline training.

## Download More Diverse Open X Episodes

Install optional Open X dependencies if needed:

```bash
uv sync --extra openx
```

Build the diverse subset:

```bash
uv run python scripts/build_diverse_openx_subset.py --overwrite
```

The intended diverse mix is:

```text
berkeley_autolab_ur5: 1000
roboturk: 1500
stanford_kuka_multimodal_dataset_converted_externally_to_rlds: 1250
bridge: 1250
```

Monitor progress:

```bash
find data/raw_diverse -maxdepth 1 -type d -name "episode_*" | wc -l
du -sh data/raw_diverse
```

## Upload Custom Open X Episode Shards To Hugging Face

Only upload custom Open X episode-level shards. Do not upload LIBERO unless explicitly requested.

Prepare upload directory:

```bash
mkdir -p hf_upload/openx_diverse_5k
cp -R data/episode_shards_diverse_5k/shards hf_upload/openx_diverse_5k/
cp -R data/episode_shards_diverse_5k/splits hf_upload/openx_diverse_5k/
cp data/episode_shards_diverse_5k/manifest.jsonl hf_upload/openx_diverse_5k/
```

Upload:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf upload-large-folder \
  Alcatraz1412/vla-franka-subset \
  hf_upload \
  --repo-type dataset \
  --include "openx_diverse_5k/**" \
  --num-workers 1
```

Download on RunPod:

```bash
uv run hf download Alcatraz1412/vla-franka-subset \
  --repo-type dataset \
  --local-dir data/hf \
  --include "openx_diverse_5k/**"
```

## Suggested First Run Order On RunPod

Run this order before any expensive experiment:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long

uv run python train.py --config configs/libero_long_sliding_window.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window.yaml

uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml

uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

## Three-Day Execution Plan

Use this plan when the goal is to produce first milestone results quickly. The target is not to finish the whole research project. The target is a credible first result table:

```text
sliding_window
event_gated_memory
age_gated_memory
event_gated_concat_query
```

All runs should use the same LIBERO-Long data split, same `K_recent`, same `H_action`, and fixed seed `42`.

### Day 1: Make The Pipeline Reliable

Goal:

```text
LIBERO data downloaded
LIBERO inspection works
smoke test passes
sliding_window starts training
event_gated_memory starts training
evaluation writes CSV rows
```

Commands:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
```

If inspection or smoke test fails, stop and fix the data or shape issue before training.

Then run the first two models:

```bash
uv run python train.py --config configs/libero_long_sliding_window.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window.yaml

uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml
```

Check:

```bash
cat results/baselines.csv
```

By the end of Day 1, there should be at least one completed result row for `sliding_window` and one for `event_gated_memory`.

### Day 2: Run The Baseline Table

Goal:

```text
complete the four-row offline LIBERO-Long baseline/ablation table
```

Run:

```bash
uv run python train.py --config configs/libero_long_sliding_window.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window.yaml

uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml

uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

The result table should have this shape:

```text
model                  gate       query             MSE     MAE     notes
sliding_window         none       none              ...
event_gated_memory     event      cross_attention   ...
age_gated_memory       age_based  cross_attention   ...
event_gated_concat     event      concat            ...
```

Do not add SmolVLA, Octo, RB-VLA, simulator rollouts, or large Open X pretraining during this three-day run unless the four-row table is already complete.

### Day 3: Interpret And Decide Whether To Pivot

Goal:

```text
decide whether the memory idea is working, weak, or needs a pivot
prepare a clean table and short written interpretation
```

Good signs:

```text
training loss decreases
validation loss is finite
event_gated MSE < sliding_window MSE
event_gated MAE < sliding_window MAE
event_gated beats age_gated
cross_attention beats concat
```

Strong result:

```text
event_gated improves MSE by roughly 5-10% or more
```

Weak but usable result:

```text
event_gated improves by 1-3%, or matches sliding_window but ablations show useful trends
```

Bad result:

```text
event_gated is much worse than sliding_window or unstable
```

If event-gated memory beats sliding window, the story is:

```text
Long-horizon memory improves offline action prediction on LIBERO-Long.
Event-aware hierarchical compression is a promising direction.
```

If event-gated memory is similar to sliding window, the story is:

```text
Offline action prediction may be too local, or the current memory module is not selective enough.
The benchmark and pipeline are ready, but stronger memory/task design is needed.
```

If event-gated memory is worse, check:

```text
training loss decreases
eval MSE is finite
state/action dims are correct
both models use the same split
event model is not undertrained relative to sliding_window
AMP is not causing instability
```

Possible pivot if event gating fails:

```text
Use age-gated hierarchical memory as the first stable memory baseline.
Treat learned/event gating as future work.
```

### Allowed Three-Day Iterations

Change only one thing at a time.

If GPU utilization is low, try:

```yaml
training:
  batch_size: 32
data:
  num_workers: 8
  prefetch_factor: 4
```

If VRAM allows, try:

```yaml
training:
  batch_size: 64
```

If training is too slow, first run shorter jobs:

```yaml
training:
  epochs: 1
  max_steps_per_epoch: 5000
```

Then scale back up after the path works.

If loss becomes NaN, try:

```yaml
training:
  lr: 5.0e-5
  amp: false
```

If event memory underperforms, try only one memory change:

```yaml
memory:
  max_memory_tokens: 8
```

or:

```yaml
memory:
  max_memory_tokens: 32
```

Do not run broad sweeps during the three-day milestone.

### Minimum Success By Sunday

Minimum deliverables:

```text
[ ] LIBERO inspection passed
[ ] LIBERO smoke test passed
[ ] sliding_window train/eval completed
[ ] event_gated_memory train/eval completed
[ ] age_gate ablation completed
[ ] concat_query ablation completed
[ ] results/baselines.csv exists
[ ] one clean result table created
[ ] short interpretation written
```

This is enough for a serious first milestone. Simulator success-rate rollouts, RB-VLA, and broader ablations come after this.

## What Counts As A Paper Baseline Right Now

Acceptable first offline baseline table:

```text
LIBERO-Long validation action MSE / MAE
sliding_window
event_gated_memory
age_gated_memory
event_gated_concat_query
```

Not enough yet for final robotics paper claim:

```text
success rate
rollout consistency
sim task completion
generalization across task suites
multi-seed confidence intervals
```

Those require simulator rollout integration and should be added after the offline pipeline is reliable.

## Next Work After First Milestone

Once the above runs are stable:

1. Add LIBERO simulator rollout scaffold.
2. Compute task success rate with a policy wrapper.
3. Add RB-VLA-style recursive belief baseline.
4. Add compression boundary ablation.
5. Add compression ratio ablation.
6. Add memory content ablation:
   - observation-only memory
   - action-only memory
   - observation-action-state memory
7. Add multi-seed runs.
8. Produce publication-ready plots and tables.

Do not start these until the sliding-window vs event-gated table is reproducible.

## Current Online Rollout State

The LIBERO online rollout stack is intentionally isolated from the main project environment because LIBERO/robosuite dependencies conflict with the main training stack.

Current isolated rollout environment:

```text
rollout venv: /workspace/libero_rollout_envs/.venv
LIBERO source: /workspace/libero_rollout_envs/LIBERO
LIBERO config: libero_config/config.yaml
LIBERO data: /workspace/vla-temporal-compression-baseline-data/libero_long
repo symlink: data/libero_long -> /workspace/vla-temporal-compression-baseline-data/libero_long
```

Use the wrapper instead of activating the rollout venv manually:

```bash
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long_sliding_window_50ep/sliding_window/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_sliding_window_50ep \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_sliding_window_50ep.csv
```

Rollout environment notes:

- `libero_rollout_env/bootstrap.sh` sets `UV_CACHE_DIR=/workspace/uv-cache` to avoid filling `/root`.
- The bootstrap path uses `hf download`, not deprecated `huggingface-cli download`.
- The rollout env has `future` installed because older LIBERO/robosuite imports need it.
- Headless rendering uses `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
- `evaluation/libero_rollout.py` patches LIBERO init-state loading with `torch.load(..., weights_only=False)` because PyTorch 2.6+ defaults to restricted loading.
- Video logging is supported with `--video-dir`, `--video-every`, and `--video-fps`.

Verified simulator wiring:

- Official `OffScreenRenderEnv` reset and zero-action stepping work.
- A LIBERO HDF5 demonstration replay on task `5` (`STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`) succeeded with reward `1` at step `222`.
- Demo replay video path: `results/rollout_videos_full/demo_replay/task05_demo0_replay.mp4`.
- Initial simulator image and HDF5 image are close enough to confirm task/action/env wiring is basically correct.

Current online rollout caveat:

- Seed-42 offline checkpoints from the earlier setup produced `0/10` successes on `libero_10` first-init rollouts.
- Corrected one-epoch sliding-window diagnostic checkpoints also produced `0/1` success on task `5`.
- Do not interpret these as final online baseline numbers yet. They were diagnostics to find training/rollout mismatches.

## Current Fixed Training Semantics

The unified episode loader was corrected to match online rollout semantics.

For timestep `t`:

```text
recent_obs      includes the current observation at t
recent_actions  contains previous executed actions only
target_actions  starts at action[t]
```

This matters because the previous loader predicted `action[t]` from observations only up to `t-1`, while online rollout predicts from the current simulator observation. The previous action-history path also leaked target/current actions into `recent_actions`; that is now fixed.

Other current training/eval fixes:

- `training/train.py` and `evaluation/eval.py` pass `recent_actions` into non-memory sliding-window models when enabled.
- `models/vla_baseline.py` supports optional `use_action_history`.
- `models/vla_baseline.py` supports optional deterministic language/task conditioning through hashed language IDs.
- `utils/language.py` contains the deterministic language hashing helper.
- `evaluation/libero_rollout.py` clips actions by default and discretizes the gripper action to `-1/+1` by default.
- `evaluation/libero_rollout.py` replans every simulator step by default with `--execute-horizon 1`.

The current `configs/libero_long_sliding_window.yaml` is the corrected sliding-window config and includes:

```yaml
model:
  use_action_history: true
  use_language: true
  language_vocab_size: 1024

training:
  gripper_loss_weight: 5.0

data:
  episode_loader:
    samples_per_epoch: 50000
```

Do not compare the old pre-fix offline MSE directly against new fixed-loader MSE. The old action-history setup had target leakage and a train/rollout temporal mismatch.

## Current 50-Epoch Phase-1 Run

The current isolated test is to train the corrected sliding-window baseline longer and change nothing else.

Important current run status as of 2026-05-20:

```text
run directory: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window
active config: logs/libero_long_sliding_window_10ep_fixed_resume_best_to50.yaml
log: logs/sliding_window_10ep_fixed_resume_best_to50.log
10-epoch completed log: logs/sliding_window_10ep_fixed.log
training stopped cleanly enough for resume after epoch 12
best checkpoint so far: best.pt from epoch 12, val_mse 0.009648240703557218
latest checkpoint: last.pt from epoch 12, val_mse 0.009648240703557218
```

The 50-epoch continuation was intentionally resumed from the intact `best.pt`, not `last.pt`,
because a short dry-run accidentally advanced `last.pt` earlier without improving `best.pt`.
That dry-run did not overwrite `best.pt`.

To stop safely after an epoch boundary, wait until the log prints an epoch summary and `last.pt`
updates, then interrupt the training process group:

```bash
pgrep -af 'sliding_window_10ep_fixed_resume_best_to50|train.py --config'
kill -INT -<process_group_id>
```

The 2026-05-20 run was stopped after epoch 12. A normal interrupt did not stop the detached
process, so it was terminated after `last.pt` and `best.pt` had both been written for epoch 12.
Re-check with `pgrep` before killing any future session.

To resume later from the latest epoch checkpoint, use:

```bash
uv run python train.py \
  --config logs/libero_long_sliding_window_10ep_fixed_resume_best_to50.yaml
```

To resume from the best-performing checkpoint instead, point the resume field in the generated config to:

```text
checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
```

After training or stopping, evaluate and run the visual task-5 rollout:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_sliding_window_50ep_fixed \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_sliding_window_50ep_fixed.csv
```

Decision rule for this phase:

```text
If success rate improves from 0% to even roughly 10-20%, keep scaling the corrected baseline.
If 50 epochs still gives roughly 0% success, then discuss architecture/data/rollout changes.
```

## RunPod Artifact Backup

Before terminating a non-persistent pod, save the artifacts that cannot be redownloaded:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
```

This packages:

```text
checkpoints/
logs/
results/
configs/
splits/
AGENTS.md
experimentation.md
libero_rollout_env/
libero_config/
utils/language.py
```

The backup includes videos under `results/rollout_videos*`. It intentionally excludes
`data/libero_long` because LIBERO should be redownloaded from the official Hugging Face repo.
If needed, upload the tarball to a private Hugging Face dataset:

```bash
uv run hf repo create Alcatraz1412/vla-run-backups --repo-type dataset --private
uv run hf upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset
```

Current uploaded backup:

```text
local tarball: /workspace/run_backups/vla_run_artifacts_20260520_190421.tar.gz
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/8ffb6186966ce8bc0644607653145e7caad3b20b
```

GitHub should receive code, configs, and documentation only. Do not commit checkpoints, logs,
videos, or LIBERO data.

## 2026-05-21 Sliding-Window Phase-1 Completion

The corrected 50-epoch sliding-window continuation finished on 2026-05-21.

Final state:

```text
run directory: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window
config: logs/libero_long_sliding_window_10ep_fixed_resume_best_to50.yaml
live log: logs/sliding_window_10ep_fixed_resume_best_to50_live.log
best checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
best checkpoint epoch: 18
best checkpoint val_mse: 0.008474992022716574
last checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/last.pt
last checkpoint epoch: 50
last checkpoint val_mse: 0.015304431917944126
epoch 50 train_mse: 0.005943
epoch 50 val_mse: 0.015304
```

Interpretation:

```text
Training loss kept decreasing, but validation did not improve after epoch 18.
The longer 50-epoch run overfit relative to the best checkpoint.
Use best.pt, not last.pt, for evaluation and comparisons.
```

Offline eval on `best.pt` completed:

```text
timestamp: 2026-05-21T12:56:40.630447+00:00
baseline: sliding_window
seed: 42
mse: 0.059324943327478
mae: 0.2940476749624525
pred_temporal_smoothness: 0.0038323509423727436
results row: results/baselines.csv
```

Task-5 online rollout on `best.pt` completed:

```text
timestamp: 2026-05-21T13:05:51.591084+00:00
task_id: 5
task_name: STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
episodes: 1
successes: 0
success_rate: 0.0
steps: 300 / 300
results csv: results/libero_rollouts_sliding_window_50ep_fixed.csv
video: results/rollout_videos_sliding_window_50ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Important environment note:

```text
The restored pod initially did not have /workspace/libero_rollout_envs/.venv.
The rollout failed until `bash libero_rollout_env/bootstrap.sh` recreated the isolated LIBERO
rollout environment. The bootstrap completed successfully and wrote libero_config/config.yaml.
```

Next clean experiment order is now:

```text
1. Run event_gated_memory under the same corrected setup.
2. Evaluate event_gated_memory offline.
3. Run task-5 rollout/video for event_gated_memory best.pt.
4. Then run age-gated and concat-query ablations.
5. Do not start broad sweeps until this corrected four-row comparison is complete.
```

Recommended next-pod setup before gated-memory training:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Restore artifacts from the latest private HF backup if this is a fresh pod.
uv run hf download Alcatraz1412/vla-run-backups \
  --repo-type dataset \
  --local-dir /workspace/run_backups

# Extract the latest tarball manually, for example:
tar -xzf /workspace/run_backups/vla_run_artifacts_YYYYMMDD_HHMMSS.tar.gz -C /root/vla-temporal-compression-baseline

# Redownload LIBERO data from the official source if needed.
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long

# Recreate isolated rollout env only if /workspace/libero_rollout_envs/.venv is missing.
bash libero_rollout_env/bootstrap.sh
```

Commands for the next runs:

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

# Age-gated ablation
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

# Concat-query ablation
uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

Before terminating a non-persistent pod after this run, create and upload a new artifact backup:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
uv run hf upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset
```

Latest 2026-05-21 uploaded backup after stopping event-gated at epoch 18:

```text
local tarball: /workspace/run_backups/vla_run_artifacts_20260521_183307.tar.gz
size: 644M
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/223967d2ad4fdacd502e714b0c2951a626909e03
```

## Superseded Event-Gated 50-Epoch Resume State

The event-gated memory model was first run for the original 10-epoch config, then resumed toward
50 epochs for a fairer comparison against the corrected 50-epoch sliding-window phase.

Superseded status as of 2026-05-21:

```text
resume config: configs/libero_long_event_gated_resume_last_to50.yaml
checkpoint dir: checkpoints/libero_long/event_gated_memory
main resume log: logs/event_gated_memory_resume_last_to50_20260521_171647.log
duplicate failed log: logs/event_gated_memory_resume_last_to50_20260521_181700.log
stopped cleanly after: epoch 18
last.pt: epoch 18, val_mse 0.012434127707300442
best.pt: epoch 6, val_mse 0.00937148479611746
best_val carried by checkpoint: 0.00937148479611746
stop time/checkpoint mtime: 2026-05-21 18:30:54 UTC
```

The duplicate launch at `2026-05-21_181700` failed immediately with CUDA OOM because the real
training job was already using the GPU. It did not produce a separate checkpoint.

Do not compare this event-gated run as a finished 50-epoch result yet. It has only reached epoch
18 of the planned 50-epoch continuation. The validation score has not beaten the epoch-6 best
checkpoint so far.

To continue the event-gated 50-epoch run on a fresh pod after restoring artifacts:

```bash
cd /root/vla-temporal-compression-baseline
uv run python train.py --config configs/libero_long_event_gated_resume_last_to50.yaml
```

This config resumes from:

```text
checkpoints/libero_long/event_gated_memory/last.pt
```

Because `last.pt` is epoch 18, training should continue from epoch 19 and run until epoch 50.
After completion, evaluate both the best checkpoint and run the task-5 rollout:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_resume_last_to50.yaml \
  --checkpoint checkpoints/libero_long/event_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated_resume_last_to50.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory_50ep \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory_50ep.csv
```

Completed 10-epoch event-gated diagnostics before the resume:

```text
offline eval checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
offline eval timestamp: 2026-05-21T16:58:10.539844+00:00
offline eval MSE: 0.06560039360608373
offline eval MAE: 0.32510020051683697
task-5 rollout timestamp: 2026-05-21T16:59:43.676875+00:00
task-5 rollout success: 0/1
task-5 rollout csv: results/libero_rollouts_event_gated_memory.csv
task-5 rollout video: results/rollout_videos_event_gated_memory/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

## Current Event-Gated 50-Epoch Completion State

The event-gated memory continuation completed on 2026-05-22 on the A100 SXM pod.

Current authoritative status:

```text
resume config: configs/libero_long_event_gated_resume_last_to50.yaml
checkpoint dir: checkpoints/libero_long/event_gated_memory
training log: logs/event_gated_memory_resume_last_to50_20260521_171647.log
gpu monitor log: logs/event_gated_memory_resume_last_to50_gpu_monitor.log
best.pt: epoch 46, val_mse 0.008947615628130734
last.pt: epoch 50, val_mse 0.010168199252802879
best_val: 0.008947615628130734
best.pt mtime: 2026-05-22 05:12:54 UTC
last.pt mtime: 2026-05-22 05:32:07 UTC
```

The A100 continuation used:

```yaml
training:
  batch_size: 64
evaluation:
  batch_size: 64
data:
  num_workers: 8
  prefetch_factor: 4
```

Use `best.pt`, not `last.pt`, for offline eval, rollout, and paper-table comparisons. The run
improved past the previous epoch-6 best and then degraded slightly by epoch 50.

Latest artifact backups created before pod termination. On a fresh pod, restore the newest
`vla_run_artifacts_*.tar.gz` from `/workspace/run_backups`.

The uploaded Hugging Face dataset is `Alcatraz1412/vla-run-backups`. Restore the newest
`vla_run_artifacts_*.tar.gz`; the 2026-05-22 backups are about 645M each.

Fresh RunPod restore sequence:

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

If a fresh pod has a dangling `data/libero_long` symlink, recreate its target under
`/workspace/vla-temporal-compression-baseline-data/libero_long` before downloading LIBERO.
If Hugging Face leaves stale LIBERO lock files, remove only
`data/libero_long/.cache/huggingface/download/libero_10/*.lock` and matching incomplete files,
then rerun the official LIBERO download.

Immediate next commands before ablations:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_resume_last_to50.yaml \
  --checkpoint checkpoints/libero_long/event_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated_resume_last_to50.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory_50ep \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory_50ep.csv
```

Then run the two planned ablations:

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```


## Current State as of 2026-05-23

Models trained and best checkpoint state:

```text
sliding_window:
  checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
  best epoch: 18
  best val_mse: 0.008474992022716574
  last.pt: epoch 50, val_mse 0.015304431917944126

event_gated_memory:
  checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
  best epoch: 46
  best val_mse: 0.008947615628130734
  last.pt: epoch 50, val_mse 0.010168199252802879

age_gated_memory:
  checkpoint: checkpoints/libero_long/age_gated_memory/best.pt
  best epoch: 18
  best val_mse: 0.011460925568826497
  last.pt: epoch 30, val_mse 0.011736674699932337
  status: stopped early on 2026-05-23 because the pod was terminating

event_gated_concat_query:
  latest available checkpoint remains the earlier 10-epoch run
  checkpoint: checkpoints/libero_long/event_gated_concat_query/best.pt
  best 10-epoch offline table MSE: 0.3393707552126476
```

Offline evals completed:

```text
sliding_window best.pt: MSE 0.059324943327478, MAE 0.2940476749624525
event_gated_memory 50-epoch best.pt: MSE 0.06263986602425575, MAE 0.2735399380326271
age_gated_memory: not run after the 2026-05-23 epoch-30 stop
event_gated_concat_query: not rerun in the 50-epoch continuation stage
```

Online rollouts completed:

```text
sliding_window task 5: success 0/1
  csv: results/libero_rollouts_sliding_window_50ep_fixed.csv
  video: results/rollout_videos_sliding_window_50ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

event_gated_memory task 5: success 0/1
  csv: results/libero_rollouts_event_gated_memory_50ep.csv
  video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory: not run after the epoch-30 stop
```

Not done yet:

- Finish or accept the stopped age-gated run. It stopped at epoch 30 with best epoch 18.
- Run offline eval for age-gated best.pt.
- Run task-5 rollout for age-gated best.pt.
- Run the concat-query continuation/ablation and its eval/rollout if still needed.

Immediate next commands:

```bash
uv run python evaluation/eval.py   --config configs/ablation_gate_age.yaml   --checkpoint checkpoints/libero_long/age_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh   configs/ablation_gate_age.yaml   checkpoints/libero_long/age_gated_memory/best.pt   --tasks 5   --episodes-per-task 1   --max-steps 300   --video-dir results/rollout_videos_age_gated_memory_50ep   --video-every 1   --video-fps 20   --results-path results/libero_rollouts_age_gated_memory_50ep.csv

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

Issues this session:

- The age-gated run was still active when the session had to end; it was stopped with SIGTERM before backup.
- Age-gated validation did not improve after epoch 18 despite training loss decreasing, so use `best.pt`, not `last.pt`.
- The sandbox wrapper failed with `bwrap: No permissions to create a new namespace`; read-only and administrative commands were run with escalation.

## Current State as of 2026-05-25

Trained models and best checkpoint info:

```text
sliding_window:
  best checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
  best epoch: 18
  best val_mse: 0.008474992022716574
  last.pt: epoch 50, val_mse 0.015304431917944126

event_gated_memory:
  best checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
  best epoch: 46
  best val_mse: 0.008947615628130734
  last.pt: epoch 50, val_mse 0.010168199252802879

age_gated_memory:
  best checkpoint: checkpoints/libero_long/age_gated_memory/best.pt
  best epoch: 31
  best val_mse: 0.010890026518609375
  last.pt: epoch 50, val_mse 0.014233005978167057

event_gated_concat_query:
  best checkpoint: checkpoints/libero_long/event_gated_concat_query/best.pt
  best epoch: 35
  best val_mse: 0.009582  # training log precision
  last.pt: epoch 50, val_mse 0.013192
```

Offline evals run:

```text
sliding_window best.pt: MSE 0.059324943327478, MAE 0.2940476749624525
event_gated_memory best.pt: MSE 0.06263986602425575, MAE 0.2735399380326271
age_gated_memory best.pt: MSE 0.0762301841750741, MAE 0.25906969606876373
event_gated_concat_query best.pt: MSE 0.06707473052665591, MAE 0.2805633209645748
```

Online rollouts run:

```text
sliding_window task 5: success 0/1
  video: results/rollout_videos_sliding_window_50ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

event_gated_memory task 5: success 0/1
  video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory task 5: success 0/1
  video: results/rollout_videos_age_gated_memory_50ep/age_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

event_gated_concat_query task 5: success 0/1
  video: results/rollout_videos_concat_query_50ep/event_gated_concat_query/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Not done yet and next immediate commands:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
uv run huggingface-cli upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset

# The first four-model single-seed offline table and task-5 diagnostic rollouts are complete.
# Next research steps are multi-task rollouts and/or multi-seed verification.
```

Bugs and issues encountered this session:

```text
The age-gated continuation initially exhausted the pod cgroup at batch_size 64 with data.num_workers 8 and prefetch_factor 4. The failure was confirmed by memory.events showing oom_kill increments and was fixed by reducing queue depth to num_workers 2/prefetch_factor 1, then increasing to the stable middle ground of num_workers 4/prefetch_factor 2.

A plain detached nohup launch was reaped by the command runner before GPU allocation. The successful detached run used nohup setsid ... so it survived the shell session and completed epoch 50.

The isolated LIBERO rollout environment has a PyTorch CUDA build without support for the RTX PRO 4500 Blackwell `sm_120` GPU. GPU rollout fails while loading model weights; age-gated and concat-query task-5 rollouts were completed with `--device cpu`.
```

## Current State as of 2026-05-31

The current active direction is no longer the older H=4 corrected-loader table. The next experiment is a corrected rollout-aligned H=1 baseline gate.

Implemented locally since the 2026-05-25 runs:

```text
evaluation/offline_diagnostics.py
scripts/compute_libero_action_stats.py
configs/libero_long_sliding_window_corrected_h1.yaml
configs/libero_long_event_gated_corrected_h1.yaml

Action normalization in the unified episode loader.
Rollout-side action unnormalization.
ImageNet normalization in training/eval/rollout paths.
Training-only augmentation in source=unified_episode.
Binary gripper loss for corrected-H1 configs.
Language conditioning for EventGatedMemoryVLA.
Masked event-gate deltas for padded older context.
Split-aware rollout init selection.
Legacy LIBERO loader warning.
```

Local LIBERO status:

```text
Full LIBERO-Long local download verified: 10 HDF5 files.
Inspection passed: 128x128 RGB, action_dim 7, state_dim 8.
Smoke test passed.
Train action stats: results/libero_action_stats_train.json
Stats demos: 400
Stats actions: 110372
```

Windows 4 GB VRAM go/no-go:

```text
Machine: RTX 3050 Laptop GPU, 4 GB VRAM, about 7.9 GB RAM
PyTorch: 2.5.1+cu121
CUDA available: yes
bf16 supported: yes

Bounded check:
python train.py --config configs/libero_long_sliding_window_corrected_h1.yaml --epochs 10 --max-steps-per-epoch 5

epoch 1 val_loss:  0.757886
epoch 10 val_loss: 0.745255

Additional movement check:
--epochs 2 --max-steps-per-epoch 50
epoch 1 train_loss 0.891342 val_loss 0.750228
epoch 2 train_loss 0.868831 val_loss 0.744029
```

Tiny checkpoint diagnostics:

```text
first_action_mse_per_element: 0.8214608968
position_mse:                 0.8370953549
rotation_mse:                 0.7470947452
continuous_mse:               0.7920950475
continuous_mae:               0.5755884461
gripper_sign_accuracy:        0.555
```

Interpretation:

```text
GO for bounded RunPod training.
The corrected training/eval path is wired and learning.
The tiny model is not rollout-ready.
The 4 GB Windows GPU is suitable only for smoke/debug runs, not full training.
```

Next clean experiment order:

```bash
# 1. Bounded corrected sliding-window RunPod gate.
uv run python train.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --epochs 1 \
  --max-steps-per-epoch 5000

uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt

# 2. Continue only if val_loss drops clearly below about 0.744
# and gripper_sign_accuracy improves beyond about 0.555.

# 3. Then run the event-gated corrected-H1 config with the same step budget.
uv run python train.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml \
  --epochs 1 \
  --max-steps-per-epoch 5000
```

Do not start a full multi-day sweep until the bounded corrected-H1 RunPod gate improves offline diagnostics.

## Current State as of 2026-06-01

RunPod environment:

```text
GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
Main env PyTorch: 2.11.0+cu128
LIBERO data: 10 HDF5 files present under data/libero_long
Rollout env: /workspace/libero_rollout_envs/.venv
```

Important implementation updates from the 2026-06-01 diagnostic:

```text
datasets/episode_loader.py:
  - load_older_context option, used to avoid decoding unused older images for sliding-window
  - task_filter option for single-task diagnostics
  - transition_sample_prob and transition_sample_radius for sampling near expert gripper transitions
  - gripper_transition target returned in each batch

datasets/data_loader.py:
  - passes the above options from data.episode_loader

training/train.py:
  - supports training.gripper_transition_loss_weight for bce_sign gripper loss

evaluation/rollout_alignment_checks.py:
  - compares checkpoint actions against a LIBERO HDF5 demo, including gripper transition timing
```

The first 10-epoch full-dataset corrected sliding-window checkpoint learned offline but still failed online:

```text
checkpoint: checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
best epoch: 3
best val_loss: 0.030981527268886568
eval continuous_mse: 0.04247721564024687
eval continuous_mae: 0.1220744714140892
eval gripper_sign_accuracy: 0.9958333373069763
task-5 train-init rollout: 0/5
task-2 train-init rollout: 0/3
task-0 train-init rollout: 0/3
task-5 demo-0 gripper transition accuracy: 0.0
```

Task-5 transition-aware overfit succeeded online:

```text
config: configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
best epoch: 20
best val_loss: 0.0019988442626804096
eval continuous_mse: 0.004982923693526134
eval continuous_mae: 0.04651936175797483
eval gripper_sign_accuracy: 0.9999000800313423
alignment demo-0 gripper transition accuracy: 1.0
task-5 train split rollout: 5/5
task-5 val split rollout: 2/5
task-5 test split rollout: 5/5
```

Useful rollout videos:

```text
results/rollout_videos_sliding_window_corrected_h1_task5_overfit_train_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
results/rollout_videos_sliding_window_corrected_h1_task5_overfit_test_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode6_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Interpretation:

```text
The corrected LIBERO rollout stack is not fundamentally broken.
The corrected-H1 policy can produce real online success.
The earlier 0% rollouts were likely caused by sparse gripper transition learning and closed-loop brittleness, not just lack of GPU capacity.
Task-5 overfit success does not prove full LIBERO-Long success or memory superiority.
```

Active next run:

```text
Train full LIBERO corrected-H1 sliding-window for 20 epochs with transition sampling/loss enabled.
Then run eval, offline diagnostics, and task 0/2/5 rollouts.
If sane, run all 10 tasks.
Then train corrected-H1 event-gated memory for the same 20-epoch protocol.
```

Active full-dataset corrected configs:

```text
configs/libero_long_sliding_window_corrected_h1.yaml
  run_name: sliding_window_corrected_h1_transition20
  checkpoint_dir: checkpoints/libero_long_corrected_transition20
  epochs: 20
  batch_size: 48
  load_older_context: false
  transition_sample_prob: 0.35
  transition_sample_radius: 4
  gripper_transition_loss_weight: 25.0

configs/libero_long_event_gated_corrected_h1.yaml
  run_name: event_gated_memory_corrected_h1_transition20
  checkpoint_dir: checkpoints/libero_long_corrected_transition20
  epochs: 20
  batch_size: 48
  transition_sample_prob: 0.35
  transition_sample_radius: 4
  gripper_transition_loss_weight: 25.0
```

## Current State as of 2026-06-02

Local Mac diagnostics were run after pulling the June 1 RunPod artifacts from Hugging Face.

Artifact source:

```text
HF dataset: Alcatraz1412/vla-run-backups
archive: runpod_20260601/vla_run_artifacts_20260601_132848.tar.gz
```

Extracted locally:

```text
checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20/best.pt
checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
results/offline_diagnostics_task5.csv
results/offline_diagnostics_transition20.csv
```

Task-5 alignment comparison across demos `0,1,2,3,4,6,12,14,21,27,34,36,39,41,45`:

```text
full_transition20:
  mean continuous MSE:        0.000643
  mean continuous MAE:        0.014436
  mean gripper accuracy:      0.994818
  transition hits:            8/15
  transition accuracy mean:   0.566667
  near-transition accuracy:   0.861538

task5_overfit:
  mean continuous MSE:        0.000435
  mean continuous MAE:        0.011628
  mean gripper accuracy:      0.996526
  transition hits:            12/15
  transition accuracy mean:   0.800000
  near-transition accuracy:   0.904762
```

Full validation per-task transition diagnostic for `sliding_window_corrected_h1_transition20`:

```text
results/per_task_transition_diagnostics_transition20_val.csv

mean task continuous_mse:        0.000862
mean task continuous_mae:        0.017611
mean task gripper accuracy:      0.966178
overall transition accuracy:     101/175 = 0.577143
overall near-transition accuracy: 943/1216 = 0.775493
```

Weakest transition tasks:

```text
LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket: 5/12 = 0.416667
LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate: 8/18 = 0.444444
LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket: 4/8 = 0.500000
LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate: 7/14 = 0.500000
```

Interpretation:

```text
The full multitask model is strong on average continuous action prediction but unreliable at rare exact gripper transitions.
Overall gripper sign accuracy is not enough because non-transition frames dominate.
Task-5 overfit success proves the rollout stack and corrected-H1 interface can work.
The project is not moving away from the memory paper idea; the reactive baseline must first execute basic grasp/release timing so memory comparisons are meaningful.
```

Recommended next work:

```text
1. Strengthen task-balanced plus transition-balanced sampling for full LIBERO.
2. Track per-task transition accuracy as a primary training/eval gate.
3. If transition accuracy stays weak, add stronger task conditioning such as task embedding into the action head, FiLM, task adapters, or lightweight task-specific heads.
4. Train event-gated memory only after applying the same improved sampling/conditioning protocol to sliding-window.
5. If improved sampling/conditioning still gives 0% rollout, pivot to ACT/action chunking before considering diffusion.
```

## Current ACT Chunking Phase As Of 2026-06-02

The corrected H1 task-balanced sliding-window run completed, but online rollouts stayed at zero success:

```text
run_name: sliding_window_corrected_h1_task_balanced_transition20
best epoch: 19
best val_loss: 0.056035
continuous_mse: 0.04575852882117033
continuous_mae: 0.12684144377708434
gripper_sign_accuracy: 0.975000011920929
exact transition accuracy: 104/175 = 0.594286
task 0 train-init rollout: 0/3
task 2 train-init rollout: 0/3
task 5 train-init rollout: 0/3
```

Task-5 trace diagnostics show closed-loop drift before grasp:

```text
episode 0: first positive gripper action 31 steps late, 0.086 m from expert grasp pose
episode 1: first positive gripper action 78 steps late, 0.139 m from expert grasp pose
episode 2: first positive gripper action 68 steps late, 0.116 m from expert grasp pose
```

Do not spend more GPU on extra H1 sliding-window epochs for this phase. The active next baseline is ACT/action chunking before diffusion and before event-gated-memory retraining.

Implemented ACT files:

```text
configs/libero_long_act_chunked_corrected_h20.yaml
models/vla_baseline.py -> baseline: act_chunked
evaluation/libero_rollout.py -> --temporal-ensemble and --trace-path
evaluation/compare_rollout_trace_to_demo.py
evaluation/per_task_transition_diagnostics.py
```

Current training command:

```bash
uv run python train.py --config configs/libero_long_act_chunked_corrected_h20.yaml
```

Current log:

```text
logs/act_chunked_corrected_h20_task_balanced_transition20_20260602.log
```

The first ACT launch exposed a dataloader transition-sampling bug for long chunks. `datasets/episode_loader.py` now filters transition anchors so the sampling window intersects valid target starts for `H_action=20`.

After ACT training completes, evaluate and run task-5 rollout:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_act_chunked_corrected_h20.yaml \
  --checkpoint checkpoints/libero_long_corrected_act_chunked_h20/act_chunked_corrected_h20_task_balanced_transition20/best.pt

bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_act_chunked_corrected_h20.yaml \
  checkpoints/libero_long_corrected_act_chunked_h20/act_chunked_corrected_h20_task_balanced_transition20/best.pt \
  --tasks 5 \
  --episodes-per-task 3 \
  --max-steps 300 \
  --split-file splits/libero_long_train.txt \
  --temporal-ensemble \
  --video-dir results/rollout_videos_act_chunked_h20_task5 \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_act_chunked_h20_task5.csv
```

ACT H20 result:

```text
training stopped after epoch 12 due validation overfitting
best epoch: 4
best val_loss: 0.291931
epoch 12 val_loss: 0.560341
task 5 rollout with temporal ensembling: 0/3
```

ACT H20 trace comparison against task-5 demos:

```text
episode 0: first positive gripper action 11 steps late, 0.0216 m from expert grasp pose
episode 1: first positive gripper action 8 steps early, 0.0366 m from expert grasp pose
episode 2: first positive gripper action 9 steps late, 0.0445 m from expert grasp pose
```

Interpretation:

```text
ACT H20 greatly improved closed-loop grasp timing and pose error compared with sliding-window but still failed success.
Do not rerun the same ACT config for more epochs.
Next inspect ACT task-5 videos and run a smaller/regularized or task-5-focused ACT diagnostic before scaling ACT or adding event-gated memory.
```

## 2026-06-03 ACT Task-5 Overfit Update

The task-5-focused ACT diagnostic has now been completed and is the current lead result.

Config:

```text
configs/libero_long_act_chunked_corrected_h20_task5_overfit.yaml
```

Training outcome:

```text
log: logs/act_chunked_corrected_h20_task5_overfit_20260603_103917.log
checkpoint dir: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit
completed epoch: 20
epoch 20 val_loss: 0.015803
best checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit/best.pt
```

Offline eval:

```text
results file: results/baselines_corrected_task5.csv
continuous_mse: 0.029388979223370554
continuous_mae: 0.12436646746397019
gripper_sign_accuracy: 0.9982025062561035
```

Task-5 train-init rollout on `best.pt`:

```text
results file: results/libero_rollouts_act_chunked_h20_task5_overfit.csv
trace file: results/rollout_trace_act_chunked_h20_task5_overfit_train3.csv
task 5: 1/3
episode 0: success in 147 steps
episode 1: fail in 300 steps
episode 2: fail in 300 steps
```

Trace comparison versus demos:

```text
episode 0: +16 steps vs expert first positive gripper, 0.0453 m grasp-pose error, success
episode 1: -11 steps, 0.0358 m, fail
episode 2: +12 steps, 0.0396 m, fail
```

Interpretation:

```text
This is the first ACT checkpoint with nonzero task-5 closed-loop success.
ACT is therefore the credible short-context baseline family for the next phase.
The remaining issue is not gross approach; it is post-grasp carry/place consistency.
Do not pivot back to H1 sliding-window.
```

Rollout env repair completed in the same phase:

```text
evaluation/libero_rollout.py now loads checkpoints without importing evaluation/eval.py
pyproject.toml declares braceexpand
libero_rollout_env/bootstrap.sh installs braceexpand
```

Active next run:

```text
configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
resume source: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit/last.pt
goal: extend from epoch 20 to epoch 40 and test whether task-5 train-init rollout can move from 1/3 to 3/3
```

## 2026-06-03 ACT Task-5 Placement Diagnostics

The epoch-40 task-5 ACT continuation improved offline continuous action error but did not improve closed-loop consistency.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt
offline continuous_mse: 0.027569980311393738
offline continuous_mae: 0.12052609633207322
gripper_sign_accuracy: 0.9987725071907043
task-5 train-init rollout: 1/3
```

The state-action/proprio-only ACT diagnostic was added to test whether robot proprio and action history were enough without vision.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_state_action.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_state_action/best.pt
offline continuous_mse: 0.11776154580116271
offline continuous_mae: 0.24411540160179138
gripper_sign_accuracy: 0.9767350002288818
task-5 train-init rollout: 0/3
```

Interpretation:

```text
Proprio/action history alone is worse than vision ACT.
Vision is helping, but the current ACT policy is not robust enough in placement.
The failure is not explained by a missing rollout dependency or pure current-state/proprio control.
```

No-training expert-prefix handoff diagnostics were added to `evaluation/libero_rollout.py` with `--expert-prefix-steps`.
They replay matching demonstration actions from the selected train-split HDF5 demo, update the online history, then hand control to ACT.

```text
normal ACT task-5 consistency40: 1/3
expert prefix 90:  1/3
expert prefix 130: 2/3
expert prefix 160: 1/3
```

Result files:

```text
results/libero_rollouts_act_chunked_h20_task5_prefix90.csv
results/libero_rollouts_act_chunked_h20_task5_prefix130.csv
results/libero_rollouts_act_chunked_h20_task5_prefix160.csv
results/rollout_trace_act_chunked_h20_task5_prefix90.csv
results/rollout_trace_act_chunked_h20_task5_prefix130.csv
results/rollout_trace_act_chunked_h20_task5_prefix160.csv
```

Current conclusion:

```text
The videos and handoff diagnostic agree: task 5 is failing mainly around placement/caddy insertion and recovery.
Prefix 130 helps because it hands off near the useful placement approach state.
Prefix 160 is not reliably better, so final contact/release behavior is also fragile.
Do not spend the next run on event memory or generic longer training.
The next useful work is a placement-focused controller/model change: phase-conditioned ACT, placement-window oversampling/loss, or a separate placement/refinement head.
```

## 2026-06-03 Late State: Placement Weighting And Diffusion

Placement-window oversampling/loss weighting was tested as the fastest placement-focused ACT change.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_placement_weighted55.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_placement_weighted55/best.pt
best epoch: 51
offline continuous_mse: 0.018447628365457058
offline continuous_mae: 0.10044001704454422
gripper_sign_accuracy: 0.9980300048828125
task-5 train-init rollout: 1/3
```

Interpretation:

```text
Placement weighting improved offline error but did not improve closed-loop success.
Failed episodes had worse grasp-pose error, so pure loss weighting can perturb approach/grasp.
Do not repeat another weighting-only run as the next step.
```

A separate small diffusion baseline was added without changing ACT/sliding-window behavior.

```text
baseline: diffusion_policy
config: configs/libero_long_diffusion_task5_h20_small.yaml
continuation config: configs/libero_long_diffusion_task5_h20_small_to50.yaml
checkpoint: checkpoints/libero_long_corrected_task5/diffusion_task5_h20_small/best.pt
stopped epoch: 35
best val denoising loss: 0.19332740310662852
sampled-action continuous_mse: 0.4715414630909697
sampled-action continuous_mae: 0.44337508891718075
gripper_sign_accuracy: 0.9636681321710824
rollout: not run
```

Interpretation:

```text
Small diffusion fits on 24 GB VRAM and trains stably.
It is not rollout-ready yet because sampled actions are still far worse than ACT.
Next architecture work should be phase-conditioned ACT or a placement/refinement head.
Only continue diffusion if offline sampled-action quality improves substantially first.
```

Previous artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260603_161004.tar.gz
HF dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/d57c1138d2700872df6451f2f632b1db3db11a37
```

## 2026-06-04 Phase ACT Baseline And Next Memory Comparison

Phase-conditioned ACT is now the frozen task-5 controller baseline.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_phase_conditioned.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_phase_conditioned/best.pt
best epoch: 58
offline continuous_mse: 0.019623747254908085
offline continuous_mae: 0.10296388152837753
gripper_sign_accuracy: 0.9973975031852722
task-5 train10 rollout: 4/10
task-5 val5 rollout:   3/5
task-5 test5 rollout:  3/5
total:                 10/20
```

Object-signal ACT was tested under the same larger rollout protocol.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_object_signals.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_object_signals/best.pt
best epoch: 69
offline continuous_mse: 0.01562450560182333
offline continuous_mae: 0.09162709747552872
gripper_sign_accuracy: 0.9983525047302246
task-5 train10 rollout: 5/10
task-5 val5 rollout:   3/5
task-5 test5 rollout:  2/5
total:                 10/20
```

Interpretation:

```text
The object-signal heuristic is a clean negative rollout result: offline improved, rollout did not.
Do not include secured / placement_ready embeddings in the main comparison.
Keep trace instrumentation, but use phase conditioning only for the next controlled memory run.
```

Next controlled research comparison:

```text
phase ACT baseline
vs
phase + event-gated memory ACT

same task: task 5
same protocol: train10 / val5 / test5
same rollout settings unless doing explicit gripper execution ablations
```

Decision rule:

```text
>= 13/20: memory is helping; continue.
~ 10/20: memory is not the current bottleneck; report phase ACT as controller baseline.
< 10/20: memory integration is adding noise or undertraining.
```

Current artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260604_124932.tar.gz
HF dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/83ba81224f5b6918ab4ffb55c245e8dc86c45ef4
```

## 2026-06-04 Event-Gated ACT Phase-Memory Positive Result

The controlled phase ACT vs phase + event-gated memory ACT comparison completed on task 5.

```text
config: configs/libero_long_event_gated_act_h20_task5_phase_memory.yaml
baseline: event_gated_act
checkpoint dir: checkpoints/libero_long_corrected_task5/event_gated_act_h20_task5_phase_memory
best checkpoint: checkpoints/libero_long_corrected_task5/event_gated_act_h20_task5_phase_memory/best.pt
best epoch: 72
best val_loss: 0.008460610160976649
stopped at epoch: 75
```

Implementation notes:

```text
ACT controller and phase conditioning were kept.
Object-signal conditioning was dropped.
Older trajectory context is compressed into event-gated memory tokens.
LIBERO HDF5 loading now supports opt-in per-worker episode RAM cache:
  data.episode_loader.cache_episodes
  data.episode_loader.cache_max_episodes
```

Offline eval on `best.pt`:

```text
continuous_mse:        0.013256419223546981
continuous_mae:        0.08408326748609543
gripper_sign_accuracy: 0.9983475049972534
```

Controlled rollout result:

```text
phase ACT:        10/20
object-signals:   10/20
event-gated ACT:  17/20

event-gated breakdown:
  train10: 8/10
  val5:    4/5
  test5:   5/5
```

Failed event-gated ACT episodes:

```text
train ep8
train ep10
val ep45
```

Interpretation:

```text
This clears the >= 13/20 decision rule for memory helping.
The result is a real positive under the current task-5 controlled protocol.
Do not make another model change before inspecting the failed traces/videos and running a larger confirmation rollout.
```

Next confirmation:

```text
task-5 train: 20
task-5 val:   10
task-5 test:  10
```

## 2026-06-08 Current Per-Task Event-Memory State

The active protocol is now per-task ACT comparison:

```text
1. Train phase-conditioned ACT single-task on the task's demos.
2. Run split-aware online rollouts.
3. Train event-gated ACT from that task's phase ACT checkpoint.
4. Run the same split-aware rollouts.
5. Compare offline continuous action metrics and online success.
```

Task 5 confirmation:

```text
phase ACT train20 / val5 / test5:
  15/20, 4/5, 4/5 = 23/30
  held-out val+test: 8/10

event-gated ACT train20 / val5 / test5:
  20/20, 4/5, 5/5 = 29/30
  held-out val+test: 9/10

summary:
  results/task5_event_memory_confirmation_20260608.md
  results/task5_confirm_trace_case_summary.csv
```

Task 2 result:

```text
task:
  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it

phase ACT:
  config: configs/libero_long_act_chunked_corrected_h20_task2_phase_conditioned.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
  stopped epoch: 30
  best epoch: 29
  continuous_mse: 0.033583113986253736
  continuous_mae: 0.13209211629629136
  gripper_sign_accuracy: 0.984755000114441
  train10 / val5 / test5: 9/10, 2/5, 4/5 = 15/20
  held-out val+test: 6/10

event-gated ACT:
  config: configs/libero_long_event_gated_act_h20_task2_phase_memory.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/event_gated_act_h20_task2_phase_memory/best.pt
  stopped epoch: 50
  best epoch: 50
  continuous_mse: 0.022640780751407148
  continuous_mae: 0.10798589040040969
  gripper_sign_accuracy: 0.9899350004196167
  train10 / val5 / test5: 10/10, 5/5, 4/5 = 19/20
  held-out val+test: 9/10

summary:
  results/task2_event_memory_comparison_20260608.md
  results/baselines_corrected_task2.csv
```

Task 2 controls completed after the first event-gated result:

```text
phase ACT continued20:
  config: configs/libero_long_act_chunked_corrected_h20_task2_phase_continued20.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_continued20/best.pt
  same phase checkpoint warm start
  train10 / val5 / test5: 6/10, 3/5, 3/5 = 12/20
  held-out val+test: 6/10

age-gated ACT continued20:
  config: configs/libero_long_age_gated_act_h20_task2_phase_memory20.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/age_gated_act_h20_task2_phase_memory20/last.pt
  same phase checkpoint warm start
  gate_type: age_based
  train10 / val5 / test5: 3/10, 2/5, 0/5 = 5/20
  held-out val+test: 2/10

corrected held-out offline eval:
  results/task2_fixed_split_offline_eval_20260608.csv

final audit:
  results/task2_final_control_audit_20260608.md
```

Task-2 video diagnostics:

```text
split file:
  splits/libero_long_task2_video_diagnostics.txt

episodes:
  6, 7, 20, 29, 40, 41

phase videos:
  results/rollout_videos_phase_act_task2_diagnostics/act_chunked_corrected_h20_task2_phase_conditioned/

event-gated videos:
  results/rollout_videos_event_gated_act_task2_diagnostics/event_gated_act_h20_task2_phase_memory/
```

Use the task-2 videos only for qualitative inspection. They were generated as reruns after the
measured rollout table, and some outcomes changed on rerun:

```text
phase episodes 40 and 41 succeeded on video rerun but failed in the original val5 run.
event episode 20 succeeded on video rerun but failed in the original test5 run.
```

Current interpretation:

```text
Event-gated ACT is now positive on task 5 and task 2 under the per-task protocol.
Task 2 is not explained by longer phase-ACT training alone or by age/recency memory.
Corrected held-out offline eval does not explain the rollout gap, which supports the
closed-loop result as a real policy difference.
Task 2 has a tied test-only subset at 4/5 against original phase ACT, so the exact claim is
aggregate split-aware improvement, not universal held-out dominance.
Next work should either repeat the same protocol on another task or run larger/multi-seed
confirmation for tasks 2 and 5 before adding broad ablations.
```

Current artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260608_132450.tar.gz
HF dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/3ef40e83472fc207cac83303bfa969dda647995f
```
