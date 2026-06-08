# LIBERO Online Rollout Environment

Isolated `uv` environment for running trained policies in the official LIBERO simulator.
Completely separate from the main project's `uv` environment — no dependency conflicts.

## RunPod Setup (one-time)

```bash
# 1. Clone repo and sync main env (for training/eval — unchanged)
cd /root
git clone <repo-url> vla-temporal-compression-baseline
cd vla-temporal-compression-baseline
uv sync

# 2. Bootstrap the LIBERO rollout env (installs into /workspace, persistent)
bash libero_rollout_env/bootstrap.sh
```

Everything goes on `/workspace` (persistent volume) so you don't lose it on pod restarts.

If the pod does not have a persistent `/workspace` volume, back up run artifacts before
terminating the pod:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
```

Upload the produced tarball to a private Hugging Face dataset or another external store.
Do not back up `data/libero_long` by default; it is large and should be downloaded again
from `yifengzhu-hf/LIBERO-datasets`.

## Running Rollouts

```bash
# Single model, single task smoke test
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long/sliding_window/best.pt \
  --tasks 0 --episodes-per-task 1 --max-steps 50

# Full evaluation: all 4 models x 3 seeds x 10 tasks
bash libero_rollout_env/run_all_rollouts.sh

# Control episodes/steps via env vars
EPISODES_PER_TASK=20 MAX_STEPS=300 bash libero_rollout_env/run_all_rollouts.sh
```

## Environment Details

| Component | Location |
|-----------|----------|
| Isolated venv | `/workspace/libero_rollout_envs/.venv` |
| LIBERO source | `/workspace/libero_rollout_envs/LIBERO` |
| LIBERO data | `/workspace/vla-temporal-compression-baseline-data/libero_long` |
| Results CSV | `results/libero_rollouts.csv` |

## Recreate On A Fresh RunPod

```bash
cd /root/vla-temporal-compression-baseline
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1

bash libero_rollout_env/bootstrap.sh

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
```

If restoring a previous run, unpack the backup tarball from the repo root before running
eval or rollout.

## Current Checkpoint And Rollout State

As of 2026-06-01, the corrected-H1 rollout stack has produced nonzero simulator success.

Task-5 transition-aware sliding-window overfit:

```text
config: configs/libero_long_sliding_window_corrected_h1_task5_overfit.yaml
checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
task-5 train split rollout: 5/5
task-5 val split rollout: 2/5
task-5 test split rollout: 5/5
```

Useful videos:

```text
results/rollout_videos_sliding_window_corrected_h1_task5_overfit_train_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
results/rollout_videos_sliding_window_corrected_h1_task5_overfit_test_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode6_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Interpretation:

```text
The rollout environment is capable of measuring real success.
The old 0/1 task-5 rollouts should remain diagnostic-only.
The next rollout-facing comparison must use the transition-aware corrected-H1 full-dataset checkpoints.
```

As of 2026-05-31, the immediate next rollout work is gated on a corrected H=1 retraining pass.
The older H=4-style 50-epoch checkpoints below all produced 0/1 on task 5 and should not be treated
as final evidence against the corrected-H1 objective.

Corrected-H1 local validation summary:

```text
configs/libero_long_sliding_window_corrected_h1.yaml passed local CUDA training checks.
Windows 4 GB RTX 3050 Laptop GPU:
  10 x 5-step check: val_loss 0.757886 -> 0.745255
  2 x 50-step check: val_loss 0.750228 -> 0.744029
  gripper_sign_accuracy on tiny checkpoint: 0.555

Decision:
  GO for bounded RunPod training.
  Run rollout only after offline diagnostics improve on a real RunPod checkpoint.
```

Next rollout-facing order:

```text
1. Train corrected sliding-window H=1 for 2000-5000 steps on RunPod.
2. Run evaluation/offline_diagnostics.py.
3. If val_loss and gripper_sign_accuracy improve, train corrected event-gated H=1.
4. Run task-5 training-init rollout before held-out split rollouts.
5. Use --split-file for held-out rollouts only after training-init behavior is sane.
```

As of 2026-05-25:

```text
sliding_window:
  checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
  best epoch: 18
  best val_mse: 0.008474992022716574
  task-5 rollout: 0/1

event_gated_memory:
  checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
  best epoch: 46
  best val_mse: 0.008947615628130734
  offline eval: MSE 0.06263986602425575, MAE 0.2735399380326271
  task-5 rollout: 0/1
  video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory:
  checkpoint: checkpoints/libero_long/age_gated_memory/best.pt
  best epoch: 31
  best val_mse: 0.010890026518609375
  offline eval: MSE 0.0762301841750741, MAE 0.25906969606876373
  task-5 rollout: 0/1
  video: results/rollout_videos_age_gated_memory_50ep/age_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

event_gated_concat_query:
  checkpoint: checkpoints/libero_long/event_gated_concat_query/best.pt
  best epoch: 35
  best val_mse: 0.009582  # training log precision
  offline eval: MSE 0.06707473052665591, MAE 0.2805633209645748
  task-5 rollout: 0/1
  video: results/rollout_videos_concat_query_50ep/event_gated_concat_query/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

On RTX PRO 4500 Blackwell pods, run rollout policy inference on CPU because the isolated rollout environment's PyTorch CUDA build does not support `sm_120`:

```bash
bash libero_rollout_env/run_rollout.sh configs/ablation_gate_age.yaml checkpoints/libero_long/age_gated_memory/best.pt --tasks 5 --episodes-per-task 1 --max-steps 300 --video-dir results/rollout_videos_age_gated_memory_50ep --video-every 1 --video-fps 20 --results-path results/libero_rollouts_age_gated_memory_50ep.csv --device cpu
```

## Current ACT Diagnostic Usage

As of 2026-06-03, the active rollout-facing baseline is task-5 ACT/action chunking, not corrected-H1 sliding-window.

Current best task-5 ACT result:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt
normal task-5 train-init rollout: 1/3
```

The rollout script supports expert-prefix handoff diagnostics:

```bash
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml \
  checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt \
  --tasks 5 \
  --episodes-per-task 3 \
  --max-steps 300 \
  --split-file splits/libero_long_train.txt \
  --expert-prefix-steps 130 \
  --video-dir results/rollout_videos_act_chunked_h20_task5_prefix130 \
  --video-every 2 \
  --video-fps 20 \
  --results-path results/libero_rollouts_act_chunked_h20_task5_prefix130.csv \
  --trace-path results/rollout_trace_act_chunked_h20_task5_prefix130.csv
```

Observed handoff results:

```text
expert prefix 90:  1/3
expert prefix 130: 2/3
expert prefix 160: 1/3
```

Interpretation:

```text
Task 5 is failing mainly around placement/caddy insertion and recovery.
The next rollout-facing work should target placement behavior, not generic longer training or memory comparison.
```

As of the later 2026-06-03 diagnostics:

```text
placement-weighted ACT:
  config: configs/libero_long_act_chunked_corrected_h20_task5_placement_weighted55.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_placement_weighted55/best.pt
  rollout: 1/3
  interpretation: better offline error, no closed-loop improvement

small diffusion policy:
  config: configs/libero_long_diffusion_task5_h20_small.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/diffusion_task5_h20_small/best.pt
  stopped epoch: 35
  sampled-action continuous_mse: 0.4715414630909697
  rollout: not run; offline sampled actions are not good enough
```

As of 2026-06-04, phase-conditioned ACT is the frozen task-5 rollout baseline:

```text
phase ACT:
  config: configs/libero_long_act_chunked_corrected_h20_task5_phase_conditioned.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_phase_conditioned/best.pt
  task-5 train10 / val5 / test5: 4/10, 3/5, 3/5 = 10/20

object-signal ACT:
  config: configs/libero_long_act_chunked_corrected_h20_task5_object_signals.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_object_signals/best.pt
  task-5 train10 / val5 / test5: 5/10, 3/5, 2/5 = 10/20
  interpretation: offline improved, rollout did not; drop object-signal conditioning from the main comparison
```

As of the event-gated ACT run later on 2026-06-04:

```text
event-gated ACT:
  config: configs/libero_long_event_gated_act_h20_task5_phase_memory.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/event_gated_act_h20_task5_phase_memory/best.pt
  best epoch: 72
  offline continuous_mse: 0.013256419223546981
  task-5 train10 / val5 / test5: 8/10, 4/5, 5/5 = 17/20
  interpretation: clears the >=13/20 memory-helping decision rule

next:
  inspect train ep8, train ep10, val ep45
  run larger confirmation rollout: train20 / val10 / test10
```

Restore the current pod artifact backup from Hugging Face if needed:

```bash
uv run hf download Alcatraz1412/vla-run-backups \
  --repo-type dataset \
  --local-dir /workspace/run_backups \
  vla_run_artifacts_20260604_124932.tar.gz

tar -xzf /workspace/run_backups/vla_run_artifacts_20260604_124932.tar.gz \
  -C /root/vla-temporal-compression-baseline
```

## 2026-06-08 Task-2 And Task-5 Rollout Artifacts

Current per-task event-memory protocol:

```text
phase ACT single-task -> split-aware rollout
event-gated ACT warm-started from that phase checkpoint -> same split-aware rollout
```

Task 5 confirmation:

```text
phase ACT:
  train20 / val5 / test5 = 15/20, 4/5, 4/5 = 23/30
  held-out val+test = 8/10

event-gated ACT:
  train20 / val5 / test5 = 20/20, 4/5, 5/5 = 29/30
  held-out val+test = 9/10
```

Task 2 result:

```text
phase ACT:
  config: configs/libero_long_act_chunked_corrected_h20_task2_phase_conditioned.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
  train10 / val5 / test5 = 9/10, 2/5, 4/5 = 15/20

phase ACT continued20:
  config: configs/libero_long_act_chunked_corrected_h20_task2_phase_continued20.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_continued20/best.pt
  train10 / val5 / test5 = 6/10, 3/5, 3/5 = 12/20

age-gated ACT:
  config: configs/libero_long_age_gated_act_h20_task2_phase_memory20.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/age_gated_act_h20_task2_phase_memory20/last.pt
  train10 / val5 / test5 = 3/10, 2/5, 0/5 = 5/20

event-gated ACT:
  config: configs/libero_long_event_gated_act_h20_task2_phase_memory.yaml
  checkpoint: checkpoints/libero_long_corrected_task2/event_gated_act_h20_task2_phase_memory/best.pt
  train10 / val5 / test5 = 10/10, 5/5, 4/5 = 19/20
```

Final task-2 audit:

```text
results/task2_final_control_audit_20260608.md

Event-gated is not explained by:
  longer phase-ACT training,
  age/recency memory,
  or the training-time validation split issue.

Remaining caveats:
  single seed,
  small selected rollout set,
  rollout nondeterminism in diagnostic reruns,
  event-gated test-only tied with original phase ACT at 4/5.
```

Task-2 diagnostic videos:

```text
diagnostic split:
  splits/libero_long_task2_video_diagnostics.txt

episodes:
  6, 7, 20, 29, 40, 41

phase video dir:
  results/rollout_videos_phase_act_task2_diagnostics/act_chunked_corrected_h20_task2_phase_conditioned/

event-gated video dir:
  results/rollout_videos_event_gated_act_task2_diagnostics/event_gated_act_h20_task2_phase_memory/

video rerun CSVs:
  results/libero_rollouts_phase_act_task2_video_diagnostics.csv
  results/libero_rollouts_event_gated_act_task2_video_diagnostics.csv
```

Video-rerun caveat:

```text
The task-2 videos were generated after the measured rollout table. Use them for qualitative
inspection only. Phase episodes 40 and 41, and event episode 20, changed outcome on video
rerun, so the original rollout CSVs remain the measured counts.
```

## Pinned Dependencies (known-working, from experimentation.md)

| Package | Version | Why pinned |
|---------|---------|------------|
| torch | 2.6.x + cu124 | robosuite compat, RTX 4090 |
| numpy | 1.26.4 | robosuite + mujoco need numpy <2 |
| mujoco | 3.8.1 | tested working version |
| robosuite | 1.4.0 | LIBERO requirement (not 1.4.1) |
| robomimic | 0.2.0 | LIBERO transitive dep |
| bddl | 1.0.1 | LIBERO requirement |
| gym | 0.25.2 | robosuite uses old gym API |

The main project uses torch 2.11+, numpy 2.x, gymnasium, etc. — those stay untouched.
