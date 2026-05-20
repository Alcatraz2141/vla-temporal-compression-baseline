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
