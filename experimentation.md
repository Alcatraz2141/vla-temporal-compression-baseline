# LIBERO-Long Experiment Log

Date: 2026-05-19

## Scope

This log records the first LIBERO-Long milestone checkpoint runs for the offline action-prediction benchmark. These are not simulator rollout results. Success rate and rollout consistency are unavailable and are logged as `nan`.

All LIBERO runs below used:

- Dataset: `libero_long`
- Seed: `42`
- Split files: `splits/libero_long_{train,val,test}.txt`
- Train/val/test episode counts: `400 / 50 / 50`
- `K_recent: 8`
- `H_action: 4`
- `image_size: 128`
- `batch_size: 32`
- Checkpoint root: `checkpoints/libero_long`
- Results CSV: `results/baselines.csv`

## Environment And Data Setup

Commands run:

```bash
uv sync
uv add h5py hf-transfer
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2
```

The first download attempt failed because `/root` was full. LIBERO storage was moved to `/workspace` and the expected repo path was restored with a symlink:

```text
data/libero_long -> /workspace/vla-temporal-compression-baseline-data/libero_long
```

Downloaded LIBERO files:

```text
10 HDF5 files
~13G at /workspace/vla-temporal-compression-baseline-data/libero_long
```

## Data Validation

Command:

```bash
uv run python scripts/inspect_libero.py --data-root data/libero_long
```

Observed structure:

```text
root keys: ['data']
episode key: data/demo_0
obs keys: ['agentview_rgb', 'ee_ori', 'ee_pos', 'ee_states', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
image shape: (272, 128, 128, 3), dtype=uint8
action shape: (272, 7), dtype=float64
state/proprio component shapes: {'ee_pos': (272, 3), 'ee_ori': (272, 3), 'gripper_states': (272, 2)}
state/proprio dim: 8
action dim: 7
```

Smoke test command:

```bash
uv run python scripts/smoke_test.py --sources libero_long
```

Smoke test result:

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

`results/smoke_test.csv` was updated successfully.

## Code/Config Fix During Runs

The first ablation attempt exposed a checkpoint hygiene issue: all memory variants used `model.baseline: event_gated_memory`, so the ablations wrote into the same checkpoint directory and logged under the same CSV label.

Fix applied:

- Added optional `model.run_name` support in `training/train.py`.
- Added optional `model.run_name` support in `evaluation/eval.py`.
- Set unique run names:
  - `event_gated_memory`
  - `age_gated_memory`
  - `event_gated_concat_query`

This keeps `model.baseline` for model construction while using `run_name` for checkpoint directories and result labels.

## One-Epoch Pipeline Check

Commands run:

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

Valid one-epoch table:

| run | MSE | MAE | smoothness |
|---|---:|---:|---:|
| `sliding_window` | 1.3018849237 | 1.8372086968 | 0.0118505776 |
| `event_gated_memory` | 0.9090182526 | 1.6646998269 | 0.0270319972 |
| `age_gated_memory` | 0.9041032791 | 1.6649569103 | 0.0270402863 |
| `event_gated_concat_query` | 1.0873619318 | 1.7739198378 | 0.0203548767 |

One-epoch interpretation:

- Memory models started cleanly and beat strict sliding-window early.
- Cross-attention beat concat at one epoch.
- Event gate and age gate were effectively tied, with age-gated slightly better on MSE.
- This was a pipeline check, not a conclusive result.

## Ten-Epoch Iteration

The four LIBERO configs were updated from `epochs: 1` to `epochs: 10`.

Commands run:

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

Final 10-epoch results, using each run's best checkpoint:

| run | MSE | MAE | smoothness | checkpoint |
|---|---:|---:|---:|---|
| `sliding_window` | 0.7020152211 | 1.2872401135 | 0.0104442284 | `checkpoints/libero_long/sliding_window/best.pt` |
| `event_gated_memory` | 0.3470005116 | 0.8869132910 | 0.0064305834 | `checkpoints/libero_long/event_gated_memory/best.pt` |
| `age_gated_memory` | 0.3627506239 | 0.9250362856 | 0.0074263177 | `checkpoints/libero_long/age_gated_memory/best.pt` |
| `event_gated_concat_query` | 0.3393707552 | 0.8595721466 | 0.0055979254 | `checkpoints/libero_long/event_gated_concat_query/best.pt` |

Relative to sliding-window MSE:

```text
event_gated_memory:        50.57% lower MSE
age_gated_memory:          48.33% lower MSE
event_gated_concat_query:  51.66% lower MSE
```

## Current Interpretation

The 10-epoch run gives a stronger signal than the one-epoch smoke run:

- All memory variants substantially outperform strict sliding-window on validation action MSE/MAE.
- The event-gated cross-attention model beats age-gated memory by MSE and MAE in this iteration.
- The concat-query ablation is slightly best overall, so the claim that cross-attention query is superior is not supported yet.
- The result supports "memory helps" more strongly than it supports the specific event-gated cross-attention design.

This is still not conclusive enough for a final paper claim because it is one seed and offline action prediction only. It is a credible first milestone table.

## Recommended Next Checkpoint

Run the same four-model table with at least three seeds:

```text
42, 43, 44
```

Report mean and standard deviation for MSE and MAE. If the same trend holds, the next decision should be whether to:

1. Keep concat-query as the stronger current memory baseline, or
2. Improve the cross-attention/event-gate design before simulator rollout work.

## Three-Seed Iteration

The same four 10-epoch configs were run with seeds `42`, `43`, and `44`.

For seeds `43` and `44`, checkpoint roots were set explicitly so runs stayed isolated:

```bash
uv run python train.py --config <config> --seed 43 --checkpoint-dir checkpoints/libero_long_seed43
uv run python evaluation/eval.py --config <config> --seed 43 --checkpoint-dir checkpoints/libero_long_seed43

uv run python train.py --config <config> --seed 44 --checkpoint-dir checkpoints/libero_long_seed44
uv run python evaluation/eval.py --config <config> --seed 44 --checkpoint-dir checkpoints/libero_long_seed44
```

Seed `42` used the original 10-epoch checkpoint root:

```text
checkpoints/libero_long
```

Per-seed results:

| seed | run | MSE | MAE | checkpoint |
|---:|---|---:|---:|---|
| 42 | `sliding_window` | 0.7020152211 | 1.2872401135 | `checkpoints/libero_long/sliding_window/best.pt` |
| 42 | `event_gated_memory` | 0.3470005116 | 0.8869132910 | `checkpoints/libero_long/event_gated_memory/best.pt` |
| 42 | `age_gated_memory` | 0.3627506239 | 0.9250362856 | `checkpoints/libero_long/age_gated_memory/best.pt` |
| 42 | `event_gated_concat_query` | 0.3393707552 | 0.8595721466 | `checkpoints/libero_long/event_gated_concat_query/best.pt` |
| 43 | `sliding_window` | 0.7697533965 | 1.2389300891 | `checkpoints/libero_long_seed43/sliding_window/best.pt` |
| 43 | `event_gated_memory` | 0.3539994572 | 0.8886521459 | `checkpoints/libero_long_seed43/event_gated_memory/best.pt` |
| 43 | `age_gated_memory` | 0.3473190367 | 0.8840091825 | `checkpoints/libero_long_seed43/age_gated_memory/best.pt` |
| 43 | `event_gated_concat_query` | 0.3530621486 | 0.9190557940 | `checkpoints/libero_long_seed43/event_gated_concat_query/best.pt` |
| 44 | `sliding_window` | 0.6226639960 | 1.1113451294 | `checkpoints/libero_long_seed44/sliding_window/best.pt` |
| 44 | `event_gated_memory` | 0.3560707825 | 0.9266813483 | `checkpoints/libero_long_seed44/event_gated_memory/best.pt` |
| 44 | `age_gated_memory` | 0.3570317285 | 0.9416718994 | `checkpoints/libero_long_seed44/age_gated_memory/best.pt` |
| 44 | `event_gated_concat_query` | 0.3300046112 | 0.9014469555 | `checkpoints/libero_long_seed44/event_gated_concat_query/best.pt` |

Aggregate results:

| run | MSE mean | MSE std | MAE mean | MAE std |
|---|---:|---:|---:|---:|
| `sliding_window` | 0.6981442046 | 0.0736210670 | 1.2125051107 | 0.0908761299 |
| `event_gated_memory` | 0.3523569171 | 0.0047529892 | 0.9007489284 | 0.0224749573 |
| `age_gated_memory` | 0.3557004631 | 0.0078014532 | 0.9169057891 | 0.0296787122 |
| `event_gated_concat_query` | 0.3408125050 | 0.0115961843 | 0.8933582987 | 0.0305556190 |

Mean MSE improvement versus sliding-window:

```text
event_gated_memory:        49.13% +/- 5.74%
age_gated_memory:          48.62% +/- 6.11%
event_gated_concat_query:  50.93% +/- 3.62%
```

Three-seed interpretation:

- The memory advantage over sliding-window is stable across seeds `42`, `43`, and `44`.
- The strongest supported claim right now is that long-horizon memory/context improves offline LIBERO-Long action prediction.
- The specific event gate is only marginally better than age gating on average.
- Concat query is the best mean performer in this small multi-seed run, so cross-attention is not yet supported as the superior query mechanism.
- This is still offline validation only. Simulator success rate remains not evaluated.

Next checkpoint:

1. Clean `results/baselines.csv` or create a publication table file with only the valid LIBERO seed rows.
2. Decide whether to pivot the "main method" toward the best-performing concat-query variant or improve the cross-attention query before rollout work.
3. After that decision, add LIBERO simulator rollout scaffolding for the selected policy variants.

## LIBERO Simulator Rollout Scaffold

The LIBERO simulator stack was installed in a separate environment so the main offline `uv` environment remains untouched:

```text
.venv-libero -> /workspace/vla-temporal-compression-baseline-envs/.venv-libero
LIBERO source: /workspace/repos/LIBERO
```

Required rollout environment variables:

```bash
export LIBERO_CONFIG_PATH=/root/vla-temporal-compression-baseline/libero_config
export PYTHONPATH=/workspace/repos/LIBERO:/root/vla-temporal-compression-baseline
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Dependency notes:

- Official LIBERO was installed editable from `/workspace/repos/LIBERO`.
- `robosuite==1.4.0`, `bddl==1.0.1`, `gym==0.25.2`, `mujoco==3.8.1`, `numpy==1.26.4`, and PyTorch CUDA were installed only into `.venv-libero`.
- `libegl1`, `libgl1`, and `libosmesa6` were installed at the system level for headless rendering.
- LIBERO init states require trusted full unpickling under PyTorch 2.6+, so the rollout script patches `torch.load(..., weights_only=False)` before LIBERO loads init states.

Added rollout entry point:

```bash
.venv-libero/bin/python evaluation/libero_rollout.py \
  --config configs/ablation_query_concat.yaml \
  --checkpoint checkpoints/libero_long/event_gated_concat_query/best.pt \
  --tasks all \
  --episodes-per-task 1 \
  --max-steps 300 \
  --seed 42 \
  --results-path results/libero_rollouts.csv
```

The script:

- creates official LIBERO `OffScreenRenderEnv` tasks,
- maps sim observations to the offline 8D proprio vector using `eef_pos + axis_angle(eef_quat) + gripper_qpos`,
- keeps online recent and older action/state/image history,
- calls sliding-window and event-memory policies with their proper tensor inputs,
- logs per-episode rows to `results/libero_rollouts.csv`.

Smoke checks:

| run | suite | task scope | max steps | episodes | successes |
|---|---|---|---:|---:|---:|
| `sliding_window` | `libero_10` | task `0` | 20 | 1 | 0 |
| `event_gated_concat_query` | `libero_10` | task `0` | 20 | 1 | 0 |

Single-task 300-step checks:

| run | suite | task scope | max steps | episodes | successes |
|---|---|---|---:|---:|---:|
| `sliding_window` | `libero_10` | task `0` | 300 | 1 | 0 |
| `event_gated_memory` | `libero_10` | task `0` | 300 | 1 | 0 |
| `age_gated_memory` | `libero_10` | task `0` | 300 | 1 | 0 |
| `event_gated_concat_query` | `libero_10` | task `0` | 300 | 1 | 0 |

All-task 300-step checks:

| run | suite | task scope | max steps | episodes | successes | success rate |
|---|---|---|---:|---:|---:|---:|
| `event_gated_concat_query` | `libero_10` | tasks `0-9`, first init each | 300 | 10 | 0 | 0.0 |
| `sliding_window` | `libero_10` | tasks `0-9`, first init each | 300 | 10 | 0 | 0.0 |

Interpretation:

- The simulator is now wired and can execute trained PyTorch policies end to end.
- These first online runs produced zero successes. This is a real simulator result, but it is not yet a conclusive policy comparison because it uses only one init state per task and the offline policy was trained without language conditioning, action normalization verification in sim, or rollout-specific validation.
- The immediate issue is the offline-to-online gap: low offline action MSE does not imply stable closed-loop LIBERO task completion.

Recommended next checkpoint:

1. Verify action-space compatibility by comparing demo action ranges against `env.action_spec` and optionally replaying one demonstration action sequence in sim.
2. Add rollout video logging for failed episodes to inspect whether the policy is immobile, saturating, drifting, or choosing semantically wrong actions.
3. Evaluate more init states only after action replay and video diagnostics pass.

## Online Rollout Debugging And Fixed-Loader Diagnostics

Date: 2026-05-20

Scope:

This entry records the debugging work done after the first online LIBERO rollouts looked poor. The goal was to determine whether the failure was simulator wiring, action-space mismatch, train/rollout mismatch, or simply an undertrained/weak policy.

### Environment Fixes

The rollout environment was rebuilt and kept isolated from the main project `uv` environment:

```text
rollout venv: /workspace/libero_rollout_envs/.venv
LIBERO source: /workspace/libero_rollout_envs/LIBERO
data root: /workspace/vla-temporal-compression-baseline-data/libero_long
repo symlink: data/libero_long -> /workspace/vla-temporal-compression-baseline-data/libero_long
```

Fixes applied:

- `libero_rollout_env/bootstrap.sh` now exports `UV_CACHE_DIR=/workspace/uv-cache`.
- `cmake` was added to bootstrap apt packages.
- Python dependency `future` was added for legacy LIBERO/robosuite imports.
- Download command was changed from deprecated `huggingface-cli download` to `hf download`.
- `libero_config/config.yaml` was pointed at the `/workspace/libero_rollout_envs/LIBERO` paths.

Verified rollout imports:

```text
torch 2.6.0+cu124
numpy 1.26.4
mujoco 3.8.1
robosuite 1.4.0
gym 0.25.2
LIBERO import OK
```

### Simulator Wiring Checks

Reset-only and zero-action stepping succeeded in the official LIBERO `OffScreenRenderEnv`.

A stored demonstration was replayed in the official simulator:

```text
task: 5
task name: STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
demo: data/demo_0
result: success, reward 1 at step 222
video: results/rollout_videos_full/demo_replay/task05_demo0_replay.mp4
```

This was the most important wiring check: expert HDF5 actions can solve the official simulator task from the stored init state. The remaining policy failures are therefore not explained by a completely broken simulator/action interface.

### Rollout Script Improvements

`evaluation/libero_rollout.py` was extended with:

- `--video-dir`
- `--video-fps`
- `--video-every`
- `--execute-horizon`
- default action clipping
- default discrete gripper conversion to `-1/+1`
- per-row `video_path` logging

The default rollout behavior now replans every simulator step:

```text
--execute-horizon 1
```

This avoids committing to a full predicted action chunk when the policy is unstable.

### Failed Online Rollouts Before Loader Fixes

The earlier seed-42 offline checkpoints were rolled out on `libero_10`, first init state per task, `max_steps=300`.

| run | tasks | episodes | successes | success rate |
|---|---:|---:|---:|---:|
| `sliding_window` | 0-9 | 10 | 0 | 0.0 |
| `event_gated_memory` | 0-9 | 10 | 0 | 0.0 |
| `age_gated_memory` | 0-9 | 10 | 0 | 0.0 |
| `event_gated_concat_query` | 0-9 | 10 | 0 | 0.0 |

Visual rollout examples:

```text
results/rollout_videos/sliding_window/seed42_task00_episode0_LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4
results/rollout_videos_full/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Initial interpretation:

- The simulator runs were real online rollouts, not offline eval.
- They were not publishable policy comparisons because the policies were trained with mismatched offline semantics and no rollout-specific validation.

### Problems Found

Several concrete issues were found:

1. Training only saw a tiny fraction of possible windows in the older config.

```text
old effective samples per epoch: about 400 windows
possible LIBERO train windows: about 105,972
10 epochs: roughly 4,000 sampled windows, about 3.8% coverage
```

The configs now use:

```yaml
data:
  episode_loader:
    samples_per_epoch: 50000
```

2. The offline loader and online rollout had a temporal mismatch.

Old loader behavior:

```text
recent_obs ended at t-1
target_actions started at action[t]
```

Online rollout behavior:

```text
policy observes current simulator observation at t
policy predicts action[t]
```

Fixed loader behavior:

```text
recent_obs includes observation[t]
target_actions starts at action[t]
```

3. The action-history path leaked target/current actions.

The old `recent_actions` indexing could include the current target action for some windows. That improves offline metrics but is impossible online.

Fixed behavior:

```text
recent_actions contains previous executed actions only
first valid timestep has zero previous action
padded action slots are zeroed
```

4. The original sliding-window model ignored `recent_actions`.

`BaselineVLA` now supports:

```yaml
model:
  use_action_history: true
```

5. The policies were not task/language conditioned.

LIBERO-10 has multiple tasks with shared scenes/objects, so a non-language-conditioned policy is ambiguous. A lightweight deterministic language/task embedding was added:

```yaml
model:
  use_language: true
  language_vocab_size: 1024
```

Implementation detail:

```text
utils/language.py hashes normalized language/task strings with blake2b into a fixed embedding-table ID.
```

The fallback LIBERO language string from HDF5 filenames is normalized by removing `_demo`, so it matches simulator task names.

### Corrected One-Epoch Diagnostics

The corrected action-history checkpoint was trained for one full fixed-loader epoch:

```bash
uv run python train.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint-dir checkpoints/libero_long_actionhist_aligned \
  --epochs 1
```

Training result:

```text
step=500 loss=0.075691
step=1000 loss=0.062325
step=1500 loss=0.057180
epoch=1 train_mse=0.056821 val_mse=0.018143
```

Standalone eval:

| checkpoint | MSE | MAE | smoothness |
|---|---:|---:|---:|
| `checkpoints/libero_long_actionhist_aligned/sliding_window/best.pt` | 0.1270002414 | 0.4544107829 | 0.0015073604 |

Task-5 rollout:

```bash
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long_actionhist_aligned/sliding_window/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_actionhist_aligned \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_actionhist_aligned.csv
```

Result:

```text
success: 0
total_reward: 0.0
steps: 300
video: results/rollout_videos_actionhist_aligned/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

### Language-Conditioned One-Epoch Diagnostics

After adding language/task conditioning and normalizing fallback language names, a one-epoch checkpoint was trained:

```bash
uv run python train.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint-dir checkpoints/libero_long_actionhist_lang_aligned \
  --epochs 1
```

Training result:

```text
step=500 loss=0.097933
step=1000 loss=0.075020
step=1500 loss=0.066431
epoch=1 train_mse=0.065683 val_mse=0.022700
```

Standalone eval:

| checkpoint | MSE | MAE | smoothness |
|---|---:|---:|---:|
| `checkpoints/libero_long_actionhist_lang_aligned/sliding_window/best.pt` | 0.1588971147 | 0.5492449743 | 0.0016622197 |

Task-5 first-frame diagnostic:

```text
pred first action:   [ 0.099, 0.272, -0.101, -0.062, 0.068, -0.048, -0.957]
expert first action: [ 0.000, 0.000, -0.000,  0.000, 0.031, -0.004, -1.000]
```

The first-frame action is closer than the hash-mismatched language checkpoint, but still commands too much XY motion.

Task-5 rollout:

```bash
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long_actionhist_lang_aligned/sliding_window/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_actionhist_lang_aligned \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_actionhist_lang_aligned.csv
```

Result:

```text
success: 0
total_reward: 0.0
steps: 300
video: results/rollout_videos_actionhist_lang_aligned/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

### Current Interpretation

The simulator and action replay are working. The remaining failure is policy quality / closed-loop robustness, not a known environment break.

The corrected loader made offline metrics less directly comparable with earlier runs because it removed action-history leakage and fixed the train/rollout observation alignment. Old pre-fix offline MSE should not be used as evidence of online readiness.

The next clean test is Phase 1: train longer with the current fixed code and change nothing else.

## Phase 1: 50-Epoch Corrected Sliding-Window Run

Date started: 2026-05-20

Purpose:

Train the corrected sliding-window policy longer without any additional changes. This isolates whether the bug fixes plus more training time are enough to produce nonzero LIBERO online success.

Command:

```bash
uv run python train.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint-dir checkpoints/libero_long_sliding_window_50ep \
  --epochs 50
```

Planned evaluation:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint checkpoints/libero_long_sliding_window_50ep/sliding_window/best.pt

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

Decision rule:

```text
If success rate improves from 0% to roughly 10-20%, the fixes mattered and the model mostly needs more data passes.
If 50 epochs still gives roughly 0% success, then discuss changes to model capacity, data/task conditioning, action treatment, or closed-loop training/evaluation.
```

### Phase 1 Interruption / Handoff Notes

Current state as of 2026-05-20:

```text
10-epoch fixed run completed.
10-epoch log: logs/sliding_window_10ep_fixed.log
10-epoch checkpoint dir: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window
10-epoch final logged train_mse: 0.018333
10-epoch final logged val_mse: 0.011661
10-epoch standalone eval MSE: 0.07697975369436401
10-epoch standalone eval MAE: 0.3519251985209329
10-epoch task-5 rollout success: 0
10-epoch task-5 rollout video: results/rollout_videos_sliding_window_10ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
10-epoch montage: results/rollout_videos_sliding_window_10ep_fixed/task05_montage.jpg
```

Qualitative result:

```text
The 10-epoch policy is not random. It moves toward the book and makes contact, but does not
grasp/place successfully. This justified continuing Phase 1, but the rollout still failed.
```

Continuation run:

```text
active generated config: logs/libero_long_sliding_window_10ep_fixed_resume_best_to50.yaml
log: logs/sliding_window_10ep_fixed_resume_best_to50.log
resume source: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
stopped after epoch 12
latest checked best.pt: epoch 12, val_mse 0.009648240703557218
latest checked last.pt: epoch 12, val_mse 0.009648240703557218
```

Important recovery note:

```text
A short one-step-per-epoch dry run accidentally advanced last.pt earlier, but did not overwrite
best.pt. The real continuation was restarted from best.pt to avoid continuing from that tainted
last.pt. Prefer best.pt for quality comparisons unless intentionally resuming latest progress.
```

If stopping the RunPod:

```bash
# Wait for an epoch summary in the active log first.
tail -f logs/sliding_window_10ep_fixed_resume_best_to50.log

# Then interrupt the training process group after re-checking the leader PID.
pgrep -af 'sliding_window_10ep_fixed_resume_best_to50|train.py --config'
kill -INT -<process_group_id>

# Package artifacts, including checkpoints and videos.
bash scripts/backup_run_artifacts.sh /workspace/run_backups
```

The artifacts backup should be uploaded to a private Hugging Face dataset or another external
store if `/workspace` is not persistent. Do not upload LIBERO data; redownload it next time.

Backup actually created and uploaded:

```text
local tarball: /workspace/run_backups/vla_run_artifacts_20260520_190421.tar.gz
size: 3.7G
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/8ffb6186966ce8bc0644607653145e7caad3b20b
```

Why task rollout is still failing:

```text
Offline MSE is supervised open-loop action prediction. Online LIBERO rollout is closed-loop and
small action errors compound over hundreds of simulator steps. The current sliding-window BC model
has learned coarse movement toward the object, but not robust grasp timing, placement, or recovery.
This is a policy-quality/closed-loop robustness issue, not currently a known simulator wiring issue,
because expert HDF5 action replay succeeds on task 5.
```

## Phase 1 Completion: Corrected Sliding Window To 50 Epochs

Date completed: 2026-05-21

The corrected sliding-window continuation finished the planned 50 epochs.

Training command actually used:

```bash
uv run python train.py \
  --config logs/libero_long_sliding_window_10ep_fixed_resume_best_to50.yaml
```

Checkpoint summary:

```text
checkpoint dir: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window
best.pt epoch: 18
best.pt val_mse: 0.008474992022716574
last.pt epoch: 50
last.pt val_mse: 0.015304431917944126
epoch 50 train_mse: 0.005943
epoch 50 val_mse: 0.015304
```

Training interpretation:

```text
Training MSE continued to decrease through epoch 50, but validation was best at epoch 18.
The 50-epoch continuation therefore overfit after the early best checkpoint.
Use best.pt for all comparisons and rollout diagnostics.
```

Offline evaluation command:

```bash
uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window.yaml \
  --checkpoint checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
```

Offline result:

```text
timestamp: 2026-05-21T12:56:40.630447+00:00
baseline: sliding_window
seed: 42
mse: 0.059324943327478
mae: 0.2940476749624525
pred_temporal_smoothness: 0.0038323509423727436
success_rate: NaN
rollout_consistency: NaN
results file: results/baselines.csv
```

Task-5 rollout command:

```bash
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

Task-5 rollout result:

```text
timestamp: 2026-05-21T13:05:51.591084+00:00
success: 0
success_rate: 0.0
total_reward: 0.0
steps: 300 / 300
csv: results/libero_rollouts_sliding_window_50ep_fixed.csv
video: results/rollout_videos_sliding_window_50ep_fixed/sliding_window/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Rollout environment recovery:

```text
On the restored pod, /workspace/libero_rollout_envs/.venv was missing.
The first rollout command failed with:
ERROR: Rollout venv not found at /workspace/libero_rollout_envs/.venv
Running `bash libero_rollout_env/bootstrap.sh` recreated the isolated rollout environment,
cloned LIBERO, installed the dependency stack, and wrote libero_config/config.yaml.
```

Current signal:

```text
The corrected sliding-window model is a useful baseline, but longer training alone did not solve
online LIBERO success. Offline action prediction improved relative to the earlier corrected
10-epoch eval, but the task-5 closed-loop rollout still failed. This justifies running the
event-gated memory model next under the same corrected setup, but if event memory also fails
online, focus should shift to train/simulator alignment, action treatment, and closed-loop
robustness rather than just more epochs.
```

## Next Runs: Event Memory And Ablations

Run these in order on the next pod after restoring artifacts and redownloading LIBERO data.

Setup:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Restore previous artifacts if starting from a clean pod.
uv run hf download Alcatraz1412/vla-run-backups \
  --repo-type dataset \
  --local-dir /workspace/run_backups

# Extract the latest tarball into the repo root.
tar -xzf /workspace/run_backups/vla_run_artifacts_YYYYMMDD_HHMMSS.tar.gz \
  -C /root/vla-temporal-compression-baseline

# LIBERO data is intentionally not backed up; redownload from official HF.
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long

# Needed only if /workspace/libero_rollout_envs/.venv is missing.
bash libero_rollout_env/bootstrap.sh
```

Event-gated memory:

```bash
uv run python train.py --config configs/libero_long_event_gated.yaml
uv run python evaluation/eval.py --config configs/libero_long_event_gated.yaml
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_event_gated.yaml \
  checkpoints/libero_long/event_gated_memory/best.pt \
  --tasks 5 \
  --episodes-per-task 1 \
  --max-steps 300 \
  --video-dir results/rollout_videos_event_gated_memory \
  --video-every 1 \
  --video-fps 20 \
  --results-path results/libero_rollouts_event_gated_memory.csv
```

Age-gated ablation:

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml
```

Concat-query ablation:

```bash
uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

After those runs, back up artifacts before stopping the pod:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
uv run hf upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset
```

Current backup uploaded after stopping the event-gated continuation at epoch 18:

```text
local tarball: /workspace/run_backups/vla_run_artifacts_20260521_183307.tar.gz
size: 644M
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/223967d2ad4fdacd502e714b0c2951a626909e03
```

## 2026-05-21 Event-Gated Memory Resume Toward 50 Epochs

Purpose:

```text
Make the event_gated_memory comparison fairer against the corrected sliding-window run by
continuing beyond the original 10-epoch gated config toward 50 epochs.
```

Resume config added:

```text
configs/libero_long_event_gated_resume_last_to50.yaml
```

Key settings:

```text
baseline: event_gated_memory
gate_type: event
query_type: cross_attention
batch_size: 32
epochs: 50
resume: checkpoints/libero_long/event_gated_memory/last.pt
checkpoint_dir: checkpoints/libero_long
run_name: event_gated_memory
```

Run command used:

```bash
nohup uv run python train.py \
  --config configs/libero_long_event_gated_resume_last_to50.yaml \
  > logs/event_gated_memory_resume_last_to50_20260521_171647.log 2>&1 &
```

A second duplicate command was accidentally launched:

```text
logs/event_gated_memory_resume_last_to50_20260521_181700.log
```

That duplicate failed immediately with CUDA OOM while the first run was active. Only one real
training process group continued.

Training was stopped intentionally after epoch 18 because the pod had no persistent volume and
needed to be terminated. Stop point:

```text
clean stop/checkpoint boundary: epoch 18
last.pt mtime: 2026-05-21 18:30:54 UTC
last.pt epoch: 18
last.pt val_mse: 0.012434127707300442
best.pt epoch: 6
best.pt val_mse: 0.00937148479611746
best_val: 0.00937148479611746
```

Epoch summaries from the resume:

```text
epoch=11 train_mse=0.015746 val_mse=0.012057
epoch=12 train_mse=0.015205 val_mse=0.010483
epoch=13 train_mse=0.014663 val_mse=0.010579
epoch=14 train_mse=0.013609 val_mse=0.010123
epoch=15 train_mse=0.013034 val_mse=0.010789
epoch=16 train_mse=0.012281 val_mse=0.012046
epoch=17 train_mse=0.011766 val_mse=0.012270
epoch=18 train_mse=0.011674 val_mse=0.012434
```

Interpretation:

```text
Training MSE is still decreasing, but validation has not improved beyond the original epoch-6
best checkpoint. Continue to 50 epochs before judging the fair comparison. If best.pt remains
epoch 6 after epoch 50, report that event-gated memory overfit early under this setup.
```

Continue on next pod:

```bash
cd /root/vla-temporal-compression-baseline
uv run python train.py --config configs/libero_long_event_gated_resume_last_to50.yaml
```

Expected behavior:

```text
The loader should restore checkpoints/libero_long/event_gated_memory/last.pt and continue from
epoch 19 through epoch 50. Keep using best.pt for eval/rollout unless last.pt later beats it.
```

Post-completion commands:

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

Then run the ablations only after the event-gated 50-epoch continuation is complete:

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```

## 2026-05-22 Event-Gated 50-Epoch Completion

The event-gated continuation from epoch 18 to epoch 50 was completed on the A100 SXM pod.

Run metadata:

```text
config: configs/libero_long_event_gated_resume_last_to50.yaml
checkpoint dir: checkpoints/libero_long/event_gated_memory
training log: logs/event_gated_memory_resume_last_to50_20260521_171647.log
gpu monitor log: logs/event_gated_memory_resume_last_to50_gpu_monitor.log
best.pt: epoch 46, val_mse 0.008947615628130734
last.pt: epoch 50, val_mse 0.010168199252802879
artifact backup dataset: Alcatraz1412/vla-run-backups
restore rule: use the newest /workspace/run_backups/vla_run_artifacts_*.tar.gz
```

A100 throughput settings used for this completion:

```text
training.batch_size: 64
evaluation.batch_size: 64
data.num_workers: 8
data.prefetch_factor: 4
```

Interpretation:

```text
The event-gated run improved beyond the earlier epoch-6 checkpoint and achieved its best
validation MSE at epoch 46. The final epoch-50 checkpoint is worse than best.pt, so comparisons
should use best.pt.
```

The run has not yet had the post-50-epoch offline evaluation command or online task-5 rollout
run after completion. Do those first on the next pod:

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

After that, continue with the planned ablations:

```bash
uv run python train.py --config configs/ablation_gate_age.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml

uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml
```


## 2026-05-23 Event-Gated Eval, Rollout, and Age-Gated Stop

What ran this session:

```bash
uv run python evaluation/eval.py   --config configs/libero_long_event_gated_resume_last_to50.yaml   --checkpoint checkpoints/libero_long/event_gated_memory/best.pt

bash libero_rollout_env/run_rollout.sh   configs/libero_long_event_gated_resume_last_to50.yaml   checkpoints/libero_long/event_gated_memory/best.pt   --tasks 5 --episodes-per-task 1 --max-steps 300   --video-dir results/rollout_videos_event_gated_memory_50ep   --video-every 1 --video-fps 20   --results-path results/libero_rollouts_event_gated_memory_50ep.csv

uv run python train.py --config configs/ablation_gate_age.yaml
pkill -TERM -f 'train.py --config configs/ablation_gate_age.yaml'
```

Key metrics:

```text
event_gated_memory best.pt: epoch 46, val_mse 0.008947615628130734
event_gated_memory offline eval: MSE 0.06263986602425575, MAE 0.2735399380326271
event_gated_memory task-5 rollout: success 0/1, reward 0.0, steps 300/300
video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory stopped state:
  best.pt: epoch 18, val_mse 0.011460925568826497
  last.pt: epoch 30, val_mse 0.011736674699932337
```

Notable age-gated validation points:

```text
epoch=18 train_mse=0.011702 val_mse=0.011461  # best
epoch=21 train_mse=0.009689 val_mse=0.011772
epoch=29 train_mse=0.005175 val_mse=0.011634
epoch=30 train_mse=0.004973 val_mse=0.011737  # stop point
```

Config changes:

```text
configs/ablation_gate_age.yaml and configs/ablation_query_concat.yaml were already modified for the 50-epoch/A100-style continuation settings: batch_size 64, num_workers 8, prefetch_factor 4, bfloat16 AMP, and checkpoint_dir checkpoints/libero_long.
```

Hardware and timing:

```text
GPU: RunPod A100-class pod for the 2026-05-23 continuation session.
age_gated batch size: 64
age_gated epoch time: roughly 8-9 minutes per epoch from the epoch-30 stop timing.
```

Issues:

```text
The pod was about to terminate, so the active age-gated run was stopped immediately with SIGTERM at epoch 30. No train/eval/rollout process remained afterward.
The sandbox helper failed with a bwrap namespace error, so file/process commands used escalation.
```

## 2026-05-25 Age-Gated 50-Epoch Completion

What ran this session:

```bash
# resumed the age-gated continuation from the epoch-30/31 checkpoint window
uv run python train.py --config configs/ablation_gate_age.yaml

# diagnostic low-queue run to avoid the cgroup OOM
# configs/ablation_gate_age.yaml: num_workers 2, prefetch_factor 1

# stable middle-ground run that completed the continuation
# configs/ablation_gate_age.yaml: num_workers 4, prefetch_factor 2
nohup setsid bash -lc 'export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1; uv run python train.py --config configs/ablation_gate_age.yaml'
```

Key metrics observed:

```text
epoch 31 best val_mse: 0.010890026519
epoch 34 val_mse at num_workers=2 / prefetch_factor=1: 0.012528234860
epoch 35 val_mse at num_workers=4 / prefetch_factor=2: 0.016462305794
final epoch 50 val_mse: 0.014233005978
best checkpoint remained epoch 31; later epochs did not beat it
offline eval: not run yet
task-5 rollout: not run yet
```

Config changes and why:

```text
batch_size stayed at 64 and lr stayed at 2e-4.
The original 8 workers / prefetch_factor 4 queue depth caused a cgroup OOM at roughly 47 GB of charged shared memory, so the queue was reduced to 2/1 for stability.
After confirming stability, the queue was increased to 4/2 as a middle-ground throughput setting; this kept the run below the memory cap and reduced epoch time to roughly 8 minutes.
```

GPU, batch size, timing:

```text
GPU: RunPod A100-class pod
batch_size: 64
epoch time at 2/1: about 20 minutes
epoch time at 4/2: about 8 minutes
```

Issues and resolution:

```text
The age-gated run hit the pod memory cgroup limit at the original 8/4 dataloader queue depth. The failure was traced with /sys/fs/cgroup/memory.events and fixed by reducing queueing to 2/1, then moving to 4/2 once the run was stable.
The command runner reaped a plain nohup background launch, so the surviving continuation used nohup setsid to detach cleanly.
```


## 2026-05-25 Concat Completion and Final Task-5 Diagnostics

What ran this session:

```bash
uv run python train.py --config configs/ablation_query_concat.yaml
uv run python evaluation/eval.py --config configs/ablation_gate_age.yaml --checkpoint checkpoints/libero_long/age_gated_memory/best.pt
uv run python evaluation/eval.py --config configs/ablation_query_concat.yaml --checkpoint checkpoints/libero_long/event_gated_concat_query/best.pt

bash libero_rollout_env/run_rollout.sh configs/ablation_gate_age.yaml checkpoints/libero_long/age_gated_memory/best.pt --tasks 5 --episodes-per-task 1 --max-steps 300 --video-dir results/rollout_videos_age_gated_memory_50ep --video-every 1 --video-fps 20 --results-path results/libero_rollouts_age_gated_memory_50ep.csv --device cpu
bash libero_rollout_env/run_rollout.sh configs/ablation_query_concat.yaml checkpoints/libero_long/event_gated_concat_query/best.pt --tasks 5 --episodes-per-task 1 --max-steps 300 --video-dir results/rollout_videos_concat_query_50ep --video-every 1 --video-fps 20 --results-path results/libero_rollouts_concat_query_50ep.csv --device cpu
```

Final offline metrics for the fixed seed-42 comparison:

```text
model                       best epoch   val_mse        offline MSE          offline MAE
sliding_window              18           0.0084749920   0.059324943327478    0.2940476749624525
event_gated_memory          46           0.0089476156   0.062639866024256    0.2735399380326271
age_gated_memory            31           0.0108900265   0.076230184175074    0.2590696960687637
event_gated_concat_query    35           0.009582       0.067074730526656    0.2805633209645748
```

Task-5 diagnostic rollout results:

```text
sliding_window:             0/1 success
event_gated_memory:         0/1 success
age_gated_memory:           0/1 success
  video: results/rollout_videos_age_gated_memory_50ep/age_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
event_gated_concat_query:   0/1 success
  video: results/rollout_videos_concat_query_50ep/event_gated_concat_query/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Notable concat-query training metrics:

```text
epoch=35 train_mse=0.009364 val_mse=0.009582  # best in available log
epoch=41 train_mse=0.006774 val_mse=0.009648
epoch=50 train_mse=0.001909 val_mse=0.013192
```

Runtime issues and resolution:

```text
The RTX PRO 4500 Blackwell GPU reports CUDA capability sm_120, but the isolated LIBERO rollout PyTorch build does not include sm_120 kernels. Attempting GPU rollout failed during model weight loading with "no kernel image is available for execution on the device". The age-gated and concat-query task-5 diagnostic rollouts were rerun with --device cpu and completed successfully.
```

## 2026-05-31 Corrected-H1 Local Implementation And 4GB GPU Go/No-Go

Goal:

```text
Before spending more RunPod budget, check whether the corrected rollout-aligned H=1 training path is actually runnable and learning on local hardware.
```

Implemented locally:

```text
evaluation/offline_diagnostics.py
scripts/compute_libero_action_stats.py
configs/libero_long_sliding_window_corrected_h1.yaml
configs/libero_long_event_gated_corrected_h1.yaml

Action normalization for the unified episode loader.
Rollout-side action unnormalization before env.step.
ImageNet image normalization in dataset/eval/rollout paths.
Training-only augmentation for source=unified_episode.
Binary gripper loss option, enabled in corrected-H1 configs.
Language conditioning for EventGatedMemoryVLA.
Masked event-gate deltas for padded older context.
Split-aware rollout init selection and dry-run selection.
Warning for legacy data.source=libero_long loader use.
```

Local LIBERO status:

```text
Full LIBERO-Long downloaded locally: 10 HDF5 files.
Inspection passed: image 128x128x3, action_dim 7, state_dim 8.
Smoke test passed.
Train action stats written: results/libero_action_stats_train.json
Stats source: splits/libero_long_train.txt
Train demos: 400
Actions counted: 110372
Continuous normalized dims: 0,1,2,3,4,5
Gripper remains unnormalized for binary sign loss.
```

Mac sanity checks:

```text
Both corrected-H1 configs built dataloaders and produced model outputs with shape (2, 1, 7).
Tiny optimizer/backprop checks passed on CPU for both sliding_window_corrected_h1 and event_gated_memory_corrected_h1.
```

Windows 4 GB VRAM machine check:

```text
RAM: about 7.9 GB
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
PyTorch: 2.5.1+cu121
CUDA: 12.1
CUDA available: True
bf16 supported: True
```

Important Windows environment note:

```text
Use the activated venv directly with python, not plain `uv run`, unless the repo lockfile has been updated for CUDA PyTorch.
Plain `uv run` may resync the environment back to CPU PyTorch from uv.lock.
```

Bounded 10-epoch 4GB check:

```text
command shape:
python train.py --config configs/libero_long_sliding_window_corrected_h1.yaml --epochs 10 --max-steps-per-epoch 5

epoch=1  train_loss=1.148910  val_loss=0.757886
epoch=2  train_loss=1.152323  val_loss=0.754818
epoch=3  train_loss=1.140435  val_loss=0.752576
epoch=4  train_loss=1.124937  val_loss=0.751295
epoch=5  train_loss=1.113524  val_loss=0.750146
epoch=6  train_loss=1.136475  val_loss=0.749162
epoch=7  train_loss=1.083851  val_loss=0.748099
epoch=8  train_loss=1.078010  val_loss=0.747053
epoch=9  train_loss=1.056888  val_loss=0.746131
epoch=10 train_loss=1.025800  val_loss=0.745255
```

Checkpoint verified:

```text
best.pt: checkpoints/gpu_check_corrected_h1_10epoch/sliding_window_corrected_h1/best.pt
last.pt: checkpoints/gpu_check_corrected_h1_10epoch/sliding_window_corrected_h1/last.pt
saved epoch: 10
best val: 0.7452554704248905
state_dim: 8
action_dim: 7
log: results/train_corrected_h1_10epoch_gpu_check.log
```

Diagnostics on the tiny checkpoint:

```text
first_action_mse_per_element: 0.8214608968
position_mse:                 0.8370953549
rotation_mse:                 0.7470947452
gripper_sign_accuracy:        0.555

eval mse:                     5.7502262732
eval mae:                     4.4383761054
continuous_mse:               0.7920950475
continuous_mae:               0.5755884461
```

Additional 4GB training movement check:

```text
run: --epochs 2 --max-steps-per-epoch 50
epoch=1 train_loss=0.891342 val_loss=0.750228
epoch=2 train_loss=0.868831 val_loss=0.744029

VRAM observed: about 439 MiB
GPU utilization observed: up to 50%
No OOM, freeze, checkpoint-load failure, NaN/inf diagnostics, or crash.
gripper_transition_accuracy NaN is expected for H_action=1 because there are no within-chunk transitions.
```

Conclusion:

```text
GO for a bounded RunPod run.

This proves the corrected training/eval path is wired and learning. It does not prove rollout success.
The 4GB machine is only suitable for smoke/debug runs, not full training.
Do not run 10 epochs x 50000 steps on the 4GB machine; at laptop speed this can take days to weeks and is fragile.
```

Next RunPod action:

```bash
uv run python train.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --epochs 1 \
  --max-steps-per-epoch 5000

uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
```

Decision rule:

```text
Continue scaling if val_loss drops clearly below the local 4GB 0.744 baseline and gripper_sign_accuracy improves beyond 0.555.
Then run configs/libero_long_event_gated_corrected_h1.yaml for the same step budget.
Pause if validation plateaus, diagnostics are NaN/inf, or gripper remains near random after meaningful training.
```

## 2026-06-01 RunPod Corrected-H1 Rollout Diagnostic

Environment:

```text
GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
Main env PyTorch: 2.11.0+cu128
CUDA available: yes
LIBERO data: 10 HDF5 files under data/libero_long
Rollout env: /workspace/libero_rollout_envs/.venv
```

Initial full-dataset corrected sliding-window run:

```bash
uv run python train.py --config configs/libero_long_sliding_window_corrected_h1.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window_corrected_h1.yaml
uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
```

Key implementation fix before this run:

- `sliding_window` was still loading unused `older_obs`, creating a large CPU/HDF5 bottleneck and low GPU utilization.
- Added `data.episode_loader.load_older_context: false` for sliding-window only.
- Throughput improved substantially because the loader stopped decoding older images that the model ignores.

10-epoch corrected sliding-window result:

```text
checkpoint: checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
best epoch: 3
best val_loss: 0.030981527268886568
eval continuous_mse: 0.04247721564024687
eval continuous_mae: 0.1220744714140892
eval gripper_sign_accuracy: 0.9958333373069763
```

Task-5 rollout with that checkpoint:

```text
task 5 train-init rollout: 0/5
task 2 train-init rollout: 0/3
task 0 train-init rollout: 0/3
```

This showed the corrected policy was learning offline but still brittle in closed-loop control.

Rollout alignment probe added:

```text
evaluation/rollout_alignment_checks.py
```

For the 10-epoch full-dataset checkpoint on task-5 demo 0:

```text
continuous_mse_raw: 0.0009216241887770593
continuous_mae_raw: 0.0173778235912323
gripper_sign_accuracy_full_demo: 0.9957264957264957
expert_gripper_transition_count: 1
gripper_sign_accuracy_at_transitions: 0.0
```

Interpretation:

- Average gripper sign accuracy was high because most timesteps do not change gripper state.
- The model missed the rare but critical gripper transition.
- In closed-loop rollout, one missed grasp/release transition can fail the whole task even if offline MSE looks good.

Task-5 targeted diagnostic changes:

- Added `task_filter` support to `datasets/episode_loader.py`.
- Added `transition_sample_prob` and `transition_sample_radius` to oversample windows near expert gripper transitions.
- Added `gripper_transition` targets to the collated batch.
- Added `training.gripper_transition_loss_weight` to upweight BCE gripper loss on transition steps.
- Added task-5 overfit config:

```text
configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
```

Task-5 overfit run:

```bash
uv run python train.py --config configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
uv run python evaluation/eval.py --config configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml \
  --checkpoint checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt \
  --results-path results/offline_diagnostics_task5.csv
```

Task-5 overfit result:

```text
checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
best epoch: 20
best val_loss: 0.0019988442626804096
eval continuous_mse: 0.004982923693526134
eval continuous_mae: 0.04651936175797483
eval gripper_sign_accuracy: 0.9999000800313423
```

Alignment probe on task-5 demo 0 after transition-aware overfit:

```text
continuous_mse_raw: 0.0004958088393323123
continuous_mae_raw: 0.012522691860795021
gripper_sign_accuracy_full_demo: 1.0
gripper_sign_accuracy_at_transitions: 1.0
predicted_close_fraction: 0.24358974358974358
expert_close_fraction: 0.24358974358974358
```

Online rollout results:

```text
task-5 train split: 5/5 success
csv: results/libero_rollouts_sliding_window_corrected_h1_task5_overfit_train_task5.csv
video dir: results/rollout_videos_sliding_window_corrected_h1_task5_overfit_train_task5/sliding_window_corrected_h1_task5_overfit/

task-5 val split: 2/5 success
csv: results/libero_rollouts_sliding_window_corrected_h1_task5_overfit_val_task5.csv
video dir: results/rollout_videos_sliding_window_corrected_h1_task5_overfit_val_task5/sliding_window_corrected_h1_task5_overfit/

task-5 test split: 5/5 success
csv: results/libero_rollouts_sliding_window_corrected_h1_task5_overfit_test_task5.csv
video dir: results/rollout_videos_sliding_window_corrected_h1_task5_overfit_test_task5/sliding_window_corrected_h1_task5_overfit/
```

Useful videos to inspect:

```text
results/rollout_videos_sliding_window_corrected_h1_task5_overfit_train_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

results/rollout_videos_sliding_window_corrected_h1_task5_overfit_test_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode6_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Current interpretation:

```text
The LIBERO rollout stack is not fundamentally broken.
The corrected H=1 policy can produce simulator success.
The prior zero-success rollouts were likely caused by undertrained closed-loop behavior, sparse gripper transition supervision, and data-loading inefficiency rather than a pure simulator/action-interface bug.
Task-5 overfit success does not prove the full multitask benchmark is solved.
The next fair comparison is to apply the same transition-aware training to both corrected sliding-window and corrected event-gated memory, then compare eval plus rollouts.
```

Next run started/planned:

```bash
mkdir -p logs
stdbuf -oL -eL uv run python train.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  2>&1 | tee logs/sliding_window_corrected_h1_transition20.log
```

Active full-dataset corrected configs as of this entry:

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

### 2026-06-01 Full-Dataset Sliding-Window Transition20 Result

20-epoch corrected-H1 full-dataset sliding-window training completed.

```text
config: configs/libero_long_sliding_window_corrected_h1.yaml
run_name: sliding_window_corrected_h1_transition20
checkpoint_dir: checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20
log: logs/sliding_window_corrected_h1_transition20.log
best checkpoint: checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20/best.pt
best epoch: 13
best val_loss: 0.05925493258982897
last epoch: 20
last val_loss: 0.0677828005515039
```

Offline eval on `best.pt`:

```text
continuous_mse: 0.0438247038051486
continuous_mae: 0.11593457460403442
gripper_sign_accuracy: 0.9708333373069763
results csv: results/baselines_corrected.csv
```

Offline diagnostics on `best.pt`:

```text
position_mse: 0.044834493841399405
position_mae: 0.12560443930405504
rotation_mse: 0.05637822325462167
rotation_mae: 0.12436454308995355
gripper_sign_accuracy: 0.965
results csv: results/offline_diagnostics_transition20.csv
```

Quick train-init rollouts on tasks 0, 2, and 5:

```text
task 0 train split: 0/3
task 2 train split: 0/3
task 5 train split: 0/3
```

Rollout CSVs:

```text
results/libero_rollouts_sliding_window_corrected_h1_transition20_train_task0.csv
results/libero_rollouts_sliding_window_corrected_h1_transition20_train_task2.csv
results/libero_rollouts_sliding_window_corrected_h1_transition20_train_task5.csv
```

Interpretation:

```text
The task-5 overfit run proves the rollout stack and corrected-H1 action interface can succeed.
The full-dataset sliding-window transition-aware run still does not produce robust train-init rollout success.
This suggests the remaining issue is not just more epochs or transition weighting; the multitask sliding-window policy is likely under-capacity, under-conditioned, or too brittle under closed-loop compounding error.
Do not claim sliding-window online success from this run.
Before spending on a full event-gated run, inspect videos and consider whether event memory should be trained as the next comparison or whether full-dataset task-balanced/transition-balanced sampling needs strengthening first.
```
