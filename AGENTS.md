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

Latest 2026-05-21 uploaded backup:

```text
local tarball: /workspace/run_backups/vla_run_artifacts_20260521_132531.tar.gz
size: 318M
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/759e608dd8c4ebbe6e8511770980c965b8575db3
```
