# LIBERO Rollout Improvement Plan

Date: 2026-05-30

This note summarizes the current diagnosis for why offline LIBERO action-prediction metrics show signal while online LIBERO rollouts remain at zero success. The goal is to separate fixable implementation/evaluation issues from true behavior-cloning limitations before spending more GPU budget.

## 2026-05-31 Local Progress Update

The Mac/local implementation pass completed the planned P0/P1 low-risk fixes and the Windows 4 GB VRAM machine completed a final go/no-go check.

Implemented locally:

- rollout-aligned offline diagnostics in `evaluation/offline_diagnostics.py`;
- train-split LIBERO action stats in `scripts/compute_libero_action_stats.py`;
- opt-in continuous action normalization for the unified LIBERO loader;
- rollout-side action unnormalization before `env.step`;
- ImageNet image normalization in training/eval and rollout paths;
- actual training-only augmentation for the unified episode loader;
- deterministic language conditioning for `EventGatedMemoryVLA`;
- masked event-gate delta features for padded older context;
- split-aware rollout init selection and dry-run selection checks;
- legacy LIBERO loader warning to avoid accidentally using old temporal semantics;
- opt-in binary gripper loss with continuous MSE for dims `0:6`;
- corrected H=1 configs:
  - `configs/libero_long_sliding_window_corrected_h1.yaml`
  - `configs/libero_long_event_gated_corrected_h1.yaml`

Full LIBERO-Long was downloaded locally and inspected:

```text
files: 10 HDF5 files under data/libero_long
train action stats: results/libero_action_stats_train.json
train demos in stats: 400
actions in stats: 110372
state_dim: 8
action_dim: 7
```

Windows 4 GB VRAM go/no-go result:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
RAM: about 7.9 GB
PyTorch: 2.5.1+cu121
CUDA available: yes
bf16 supported: yes
LIBERO inspect: passed
LIBERO smoke test: passed
```

Bounded local training evidence:

```text
config: configs/libero_long_sliding_window_corrected_h1.yaml
run: --epochs 10 --max-steps-per-epoch 5

epoch 1  val_loss 0.757886
epoch 10 val_loss 0.745255

checkpoint: checkpoints/gpu_check_corrected_h1_10epoch/sliding_window_corrected_h1/best.pt
```

Additional 4 GB GPU check:

```text
run: --epochs 2 --max-steps-per-epoch 50
epoch 1 train_loss 0.891342 val_loss 0.750228
epoch 2 train_loss 0.868831 val_loss 0.744029
VRAM observed: about 439 MiB
GPU utilization observed: up to about 50%
```

Diagnostics on the tiny checkpoint:

```text
first_action_mse_per_element: 0.8214608968
position_mse:                 0.8370953549
rotation_mse:                 0.7470947452
continuous_mse:               0.7920950475
continuous_mae:               0.5755884461
gripper_sign_accuracy:        0.555
```

Interpretation:

- The corrected training path is no longer speculative: CUDA, data loading, loss/backprop, validation, checkpointing, eval, and diagnostics all work.
- The tiny local model is not a good policy yet. Gripper sign accuracy is only slightly above random, which is expected with tiny training.
- The local result justifies a bounded RunPod training run. It does not prove final rollout success.
- The 4 GB machine is useful for smoke/debug only; full `10 x 50000` or memory-model training is not practical there.

## Current Working Hypothesis

The simulator/action interface is basically wired because expert HDF5 replay succeeds in the official LIBERO simulator. The remaining failures are likely a mixture of:

- train/evaluation objective mismatch,
- action scaling and gripper modeling issues,
- visual preprocessing/generalization issues,
- missing task/language conditioning in memory models,
- closed-loop compounding errors from plain behavior cloning,
- and the inherent difficulty of LIBERO-Long.

Do not jump straight to broad memory sweeps or heavier architectures until the P0 diagnostics and baseline fixes are complete.

## 2026-06-01 RunPod Update

The first 24 GB RTX 4090 corrected-H1 pass found two practical issues:

- the strict sliding-window model was still paying the cost to load/decode unused older image context;
- rare gripper open/close transitions were too easy for the loss and offline metrics to miss.

Implemented fixes:

- `data.episode_loader.load_older_context: false` for sliding-window configs;
- `task_filter` support for isolated task diagnostics;
- `transition_sample_prob` / `transition_sample_radius` for sampling near expert gripper transitions;
- `gripper_transition` batch targets;
- `training.gripper_transition_loss_weight` for transition-weighted binary gripper loss;
- `evaluation/rollout_alignment_checks.py` for demo-level rollout alignment checks.

Diagnostic evidence:

```text
10-epoch full corrected sliding-window:
  eval continuous_mse: 0.04247721564024687
  eval continuous_mae: 0.1220744714140892
  gripper_sign_accuracy: 0.9958333373069763
  task-5 train-init rollout: 0/5
  demo-0 gripper transition accuracy: 0.0

20-epoch task-5 transition-aware overfit:
  eval continuous_mse: 0.004982923693526134
  eval continuous_mae: 0.04651936175797483
  gripper_sign_accuracy: 0.9999000800313423
  demo-0 gripper transition accuracy: 1.0
  task-5 train split rollout: 5/5
  task-5 val split rollout: 2/5
  task-5 test split rollout: 5/5
```

Interpretation:

```text
The rollout stack is capable of success.
The old 0% rollout result was not proof that LIBERO sim wiring or corrected-H1 training was hopeless.
The immediate bottleneck was sparse, high-impact gripper transition behavior plus closed-loop brittleness.
Full-dataset retraining is still required before comparing sliding-window against event-gated memory.
```

Current next experiment:

```text
1. Train full LIBERO corrected-H1 sliding-window for 20 epochs with transition sampling/loss enabled.
2. Eval and run offline diagnostics.
3. Roll out tasks 0, 2, and 5 first.
4. If nonzero/stable, roll out all 10 tasks.
5. Repeat the same budget and rollout protocol for corrected-H1 event-gated memory.
```

## 2026-06-02 Local Per-Task Diagnostic Update

The 20-epoch full-dataset corrected-H1 sliding-window model was diagnosed locally on the Mac using the June 1 RunPod artifact.

Artifact:

```text
runpod_20260601/vla_run_artifacts_20260601_132848.tar.gz
checkpoint: checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20/best.pt
comparison checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
```

Task-5 comparison across selected train/val/test demos:

```text
full_transition20:
  mean continuous MSE:      0.000643
  mean continuous MAE:      0.014436
  transition hits:          8/15

task5_overfit:
  mean continuous MSE:      0.000435
  mean continuous MAE:      0.011628
  transition hits:          12/15
```

Full validation split per-task diagnostic:

```text
results/per_task_transition_diagnostics_transition20_val.csv

mean task continuous_mse:        0.000862
mean task continuous_mae:        0.017611
mean task gripper accuracy:      0.966178
overall transition accuracy:     101/175 = 0.577143
overall near-transition accuracy: 943/1216 = 0.775493
```

Conclusion:

```text
Average action prediction is not the blocking issue anymore.
The blocking issue is reliable rare transition timing across all tasks.
Do not spend on a broad event-gated run until the reactive baseline uses stronger task-balanced and transition-balanced training, otherwise the memory comparison will be confounded by a weak controller.
```

Updated next experiment:

```text
1. Strengthen task-balanced plus transition-balanced sampling for full LIBERO.
2. Use per-task exact transition accuracy as a first-class metric.
3. If transition accuracy remains weak, add stronger task conditioning before memory comparisons.
4. Train corrected-H1 event-gated memory only under the same improved protocol.
5. If this still produces zero rollout success, pivot to ACT/action chunking.
```

## Priority Summary

| Priority | Issue | Why It Matters | Local On M4 Pro? | Needs GPU For Final Evidence? |
|---|---|---|---:|---:|
| P0 | Offline metric does not match deployed control action | Evaluation averages all `H_action=4` predictions, but rollout currently executes only action head `0` with `--execute-horizon 1`. | Yes | No |
| P0 | No action normalization in active unified LIBERO path | Translation/rotation deltas and gripper commands have very different scales, making optimization and control imprecise. | Yes for code/stats | Yes for retraining |
| P0 | Memory models lack language conditioning | Corrected `sliding_window` uses task/language conditioning, but `EventGatedMemoryVLA` does not. Multi-task LIBERO memory comparisons are confounded. | Yes | Yes for fair memory retraining |
| P1 | Rollout init states are not split-aware | Rollouts use init indices directly, while offline splits are by HDF5 demo IDs. Some diagnostic rollouts are on training demos, not held-out episodes. | Yes for code | Linux rollout recommended |
| P1 | ImageNet-pretrained ResNet gets unnormalized `[0, 1]` RGB | Pretrained ResNet expects ImageNet normalization. Freezing the current encoder before fixing this would be misleading. | Yes | Yes for retraining comparison |
| P1 | `augment.enabled: true` is ignored by unified loader | Configs say augmentation is enabled, but the active `source: unified_episode` loader does not apply it. | Yes | Yes for effect size |
| P1 | Gripper is trained as weighted regression | Rollout thresholds gripper to `-1/+1`, but training optimizes continuous MSE. A binary gripper objective better matches deployment. | Yes | Yes for retraining |
| P2 | Event-gate deltas include padded transitions | Delta features for padded older context can distort event scores early in trajectories. | Yes | Yes for fair memory retraining |
| P2 | Legacy LIBERO loader still has old temporal semantics | Current configs use the corrected unified loader, but accidentally switching paths can reintroduce old mismatch/leakage behavior. | Yes | No |

## P0 Details

### P0.1 Add Deployed-Action Metrics

Current offline MSE/MAE averages all predicted actions in the chunk. Online rollout replans every simulator step and executes only the first predicted action by default.

Add metrics:

- `first_action_mse`
- `first_action_mae`
- per-horizon MSE/MAE for action heads `0..H_action-1`
- per-action-dimension MSE/MAE
- position-only MSE/MAE for dims `0:3`
- rotation-only MSE/MAE for dims `3:6`
- gripper sign accuracy
- gripper transition accuracy
- metrics near contact/gripper transition windows

Expected outcome:

If action head `0` is much worse than the chunk average, current offline metrics are overestimating rollout readiness.

Local feasibility:

This is pure evaluation code and can be done locally on the MacBook, assuming checkpoints and enough LIBERO data are available. It does not require GPU.

### P0.2 Add Action Normalization

The active unified LIBERO loader currently returns raw actions. Add train-split action statistics and normalize continuous action dimensions during training.

Recommended first version:

- Compute stats from train episodes only.
- Normalize dims `0:6`.
- Do not standardize gripper as a regression target if switching to binary gripper loss.
- Store stats under `data/libero_long/action_stats.json` or `results/action_stats_libero_long_train.json`.
- Save the stats path in the config.
- Unnormalize predicted continuous actions before simulator `env.step`.
- Normalize previous-action history consistently before feeding it to the model.

Expected outcome:

Better positional/rotational control and less loss domination by high-magnitude dimensions.

Local feasibility:

Stats computation and code changes are local-friendly. Real comparison requires retraining on GPU.

### P0.3 Add Language Conditioning To Memory Models

The corrected sliding-window baseline supports:

```yaml
model:
  use_language: true
  language_vocab_size: 1024
```

`EventGatedMemoryVLA` currently ignores language/task strings. Add the same deterministic language embedding path used by `BaselineVLA`.

Expected outcome:

Fairer comparison on LIBERO-10/Long, where multiple tasks share scenes and objects. Without language conditioning, memory models may predict an averaged ambiguous policy.

Local feasibility:

Implementation and shape tests are local-friendly. Fair performance comparison requires retraining.

## P1 Details

### P1.1 Make Rollout Init Selection Split-Aware

The rollout script currently uses:

```text
suite.get_task_init_states(task_id)[episode_idx]
```

The offline splits are stored as HDF5 demo IDs like:

```text
libero_long::libero_10/..._demo.hdf5::data/demo_21
```

Add a way to run rollouts on init indices corresponding to train/val/test split files.

Recommended CLI additions:

```bash
--split-file splits/libero_long_test.txt
--split train|val|test
--episodes-per-task 20
```

Keep two rollout modes:

- training-init sanity rollouts, used to check whether the model can solve seen initial states;
- held-out split rollouts, used for actual reporting.

Expected outcome:

Cleaner interpretation. If a policy cannot solve training-init rollouts, do not expect held-out success.

Local feasibility:

The code can be written locally. Running official LIBERO simulator rollouts is still safer on Linux/RunPod because the isolated rollout environment depends on LIBERO/robosuite/MuJoCo.

### P1.2 Fix Image Preprocessing For Pretrained ResNet

The models request ImageNet-pretrained ResNet18, but the active paths feed raw `[0, 1]` tensors. Add consistent normalization in both training/eval and rollout image tensor creation.

ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Expected outcome:

More meaningful pretrained visual features and a fairer frozen/backbone-LR experiment.

Important:

Do not run a "freeze ResNet" experiment before this is fixed. Freezing an improperly normalized pretrained encoder is likely to hurt.

### P1.3 Implement Actual Unified-Loader Augmentation

The current LIBERO configs set:

```yaml
data:
  augment:
    enabled: true
```

But `source: unified_episode` does not currently apply augmentation. Add training-only augmentation similar to the older `LiberoLongDataset` path:

- `RandomResizedCrop`
- mild `ColorJitter`
- possibly small affine/crop jitter if rollout camera shift is suspected

Expected outcome:

Better visual robustness under simulator camera/rendering differences.

### P1.4 Split Gripper Into Classification

Current training uses weighted regression, then rollout hard-thresholds predicted gripper. Better alignment:

- regression loss for continuous dims `0:6`;
- binary classification loss for gripper open/close;
- optional transition weighting around gripper changes;
- rollout maps classifier output to `-1/+1`.

Expected outcome:

Cleaner grasp timing and fewer half-closed gripper actions.

## P2 Details

### P2.1 Mask Event-Gate Delta Features

`EventGatedMemoryVLA` computes visual/action/state deltas with `diff()` over chunked older context. Padded entries should not contribute to event scores.

Fix:

- compute pairwise valid masks inside chunks;
- only average deltas over valid adjacent pairs;
- handle all-padding chunks safely.

Expected outcome:

Fairer and more stable memory scoring, especially early in trajectories.

### P2.2 Guard Against Legacy Loader Reuse

The older `datasets/libero_long_dataset.py` path still contains earlier temporal semantics and action-history behavior. Current corrected configs use:

```yaml
data:
  source: unified_episode
```

Add warnings or tests to prevent accidental use of the old LIBERO loader for corrected rollout-facing experiments.

Expected outcome:

Avoid accidentally reintroducing target leakage or train/rollout timestep mismatch.

## Model And Benchmark Strategy

### Keep LIBERO-Long, But Do Not Debug Only On LIBERO-Long

LIBERO-Long is the target benchmark for the memory hypothesis, but it is a hard starting point for validating basic behavior cloning. Use easier settings as a controller sanity ladder.

Suggested ladder:

| Test | Meaning If It Fails |
|---|---|
| Expert HDF5 replay succeeds | Already passed; simulator/action interface is basically working. |
| Policy cannot solve a training init on one easy task | Model/output/preprocessing problem. |
| Policy solves training init but not held-out easy-task inits | Generalization/augmentation problem. |
| Policy succeeds on easier suite but not LIBERO-Long | Horizon/recovery/memory problem. |
| Memory improves LIBERO-Long after reactive baseline works | Stronger support for the research hypothesis. |

Official LIBERO suites to consider:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10` / LIBERO-Long

Use simpler suites diagnostically. Do not abandon LIBERO-Long as the long-horizon memory target unless the research story explicitly pivots.

## Local MacBook Pro M4 Pro Plan

The MacBook should be used for code, data inspection, metrics, and tiny smoke tests. Avoid full training locally unless it is a tiny overfit/debug run.

Good local tasks:

- add per-horizon/per-dimension metrics;
- compute action statistics from train split;
- add action normalization/unnormalization code;
- add binary gripper target/loss code;
- add ImageNet normalization in dataset and rollout;
- implement actual augmentation for unified loader;
- add language conditioning to memory models;
- mask event-gate deltas;
- add split-aware rollout index mapping;
- run smoke tests with five windows;
- run tiny overfit tests on one to five episodes;
- run `--epochs 1 --max-steps-per-epoch 20` style checks.

Avoid locally:

- full 50-epoch LIBERO training;
- four-model retraining;
- broad ablations;
- large rollout campaigns if LIBERO/robosuite is not stable on macOS.

## GPU Budget Plan

Do not spend GPU on all four models immediately. The 2026-05-31 local checks prove the corrected reactive baseline path is runnable and learning, so the next paid run should be bounded rather than a full blind sweep.

Recommended first RunPod run:

```text
model: sliding_window
H_action: 1
action normalization: on
binary gripper: on
ImageNet normalization: on
language: on
augmentation: on
budget gate: 1 epoch, 2000-5000 steps
```

Then evaluate:

- first-action offline metrics;
- per-dim metrics;
- continuous action MSE/MAE;
- gripper sign accuracy;
- training-init rollout on one easier task or task 5;
- held-out rollout only if training-init behavior is sane.

Decision rule:

- Continue scaling if validation loss drops clearly below the local 4 GB baseline around `0.744` and gripper sign accuracy improves beyond the tiny-checkpoint `0.555`.
- Stop and debug if validation loss plateaus, diagnostics become NaN/inf, or gripper sign accuracy remains near random after a meaningful number of steps.
- Only after the corrected sliding-window policy shows stable improvement should `event_gated_memory_corrected_h1` be trained under the same step budget.
- Only after the corrected sliding-window policy shows non-random closed-loop behavior should the broader memory ablation table be rerun.

## Rough GPU-Time Estimates

Assumptions:

- one 32 GB VRAM GPU, roughly RTX 4090/A5000/L40 class;
- LIBERO images at `128x128`;
- batch size `32-64`;
- 50k samples per epoch;
- dataloader is not badly IO-bound.

| Job | Rough Time |
|---|---:|
| Metrics only | minutes, no GPU needed |
| One-epoch smoke train | 10-30 minutes |
| 10-epoch corrected sliding-window | 2-5 hours |
| 20-epoch corrected sliding-window | 4-10 hours |
| 50-epoch corrected sliding-window | 10-24 hours |
| 50-epoch event-memory run | 12-30 hours |
| Four-model corrected table | roughly 2-5 GPU-days |

These estimates are intentionally conservative. Prior A100 runs were much faster per epoch, but cheaper 32 GB GPUs may be slower and more IO-bound.

## Recommended Execution Order

1. P0/P1 implementation is complete and the 2026-06-01 task-5 diagnostic proved nonzero online success.
2. The 2026-06-02 diagnostic showed full-dataset sliding-window still hits only `101/175` validation gripper transitions.
3. Strengthen task-balanced plus transition-balanced sampling before another full comparison run.
4. Evaluate with per-task transition accuracy, continuous MSE/MAE, and train-init rollouts.
5. Add stronger task conditioning if transition accuracy remains weak.
6. Train `configs/libero_long_event_gated_corrected_h1.yaml` only after the baseline uses the same improved protocol.
7. Compare sliding-window and event-gated only after both have the same corrected-H1 transition-aware treatment.
8. If improved sliding-window is still 0% despite task-5 overfit success, pivot to ACT/action chunking before diffusion.

## What Not To Do Yet

- Do not run broad memory sweeps before the corrected sliding-window baseline works.
- Do not claim LIBERO rollout success from Open X pretraining.
- Do not report old pre-fix offline metrics as comparable to corrected-loader metrics.
- Do not freeze the visual encoder before adding ImageNet normalization.
- Do not treat one failed LIBERO-Long rollout as proof that memory is useless.
- Do not treat one successful easier-suite rollout as proof that LIBERO-Long is solved.

## Success Criteria For The Next Milestone

Minimum next milestone:

- P0 diagnostics exist and are logged.
- Action stats are computed from train split only.
- Corrected `sliding_window` training starts and loss decreases.
- First-action/per-dim metrics are finite.
- At least one training-init rollout is qualitatively sane.
- If possible, at least one easier-suite or single-task rollout gets non-zero success.

Stronger milestone:

- Corrected `sliding_window` gets non-zero held-out success on an easier LIBERO suite.
- Memory models are language-conditioned and normalized identically.
- Corrected memory comparison is rerun only after the baseline controller is credible.

## 2026-06-02 Update: ACT Chunking Is Now Active

The task-balanced corrected-H1 sliding-window run completed and remained stable, but it did not solve online execution:

```text
run_name: sliding_window_corrected_h1_task_balanced_transition20
best epoch: 19
best val_loss: 0.056035
continuous_mse: 0.04575852882117033
continuous_mae: 0.12684144377708434
gripper_sign_accuracy: 0.975000011920929
exact transition accuracy: 104/175 = 0.594286
train-init rollouts task 0/2/5: 0/3, 0/3, 0/3
```

Task-5 trace diagnostics showed closed-loop drift before grasp rather than a pure offline gripper-label failure:

```text
episode 0: first positive gripper action 31 steps late, 0.086 m from expert grasp pose
episode 1: first positive gripper action 78 steps late, 0.139 m from expert grasp pose
episode 2: first positive gripper action 68 steps late, 0.116 m from expert grasp pose
```

Implementation changes now in the repo:

```text
configs/libero_long_act_chunked_corrected_h20.yaml
models/vla_baseline.py: act_chunked baseline
evaluation/libero_rollout.py: --temporal-ensemble and --trace-path
evaluation/compare_rollout_trace_to_demo.py
evaluation/per_task_transition_diagnostics.py
```

Current active training:

```bash
uv run python train.py --config configs/libero_long_act_chunked_corrected_h20.yaml
```

Current log:

```text
logs/act_chunked_corrected_h20_task_balanced_transition20_20260602.log
```

Decision rule:

```text
If ACT H20 recovers task-5 train-init grasp or substantially reduces grasp-pose error, continue with ACT as the credible short-context baseline.
Then port event-gated memory onto the chunked-action head for a fair memory comparison.
If ACT H20 still misses the book similarly, inspect action scale/temporal ensembling and consider stronger task/object conditioning before diffusion.
Do not spend more GPU on additional H=1 sliding-window epochs.
```

### ACT H20 Result

The ACT H20 run was stopped after epoch 12 because validation degraded while training loss kept decreasing:

```text
best epoch: 4
best val_loss: 0.291931
epoch 12 val_loss: 0.560341
```

Offline eval on `best.pt`:

```text
continuous_mse: 0.3000641145876476
continuous_mae: 0.38405559744153706
gripper_sign_accuracy: 0.9214285697255816
first_action_mse_per_element: 2.0423375430099435
first_action_mae_per_element: 0.7068742666134079
```

Task-5 train-init rollout with temporal ensembling:

```text
task 5: 0/3
```

Trace comparison against the same demos:

```text
episode 0: first positive gripper action 11 steps late, 0.0216 m from expert grasp pose
episode 1: first positive gripper action 8 steps early, 0.0366 m from expert grasp pose
episode 2: first positive gripper action 9 steps late, 0.0445 m from expert grasp pose
```

Updated interpretation:

```text
ACT H20 improved closed-loop timing and grasp-pose error a lot compared with sliding-window, but still got 0/3 success.
The remaining failure is probably contact geometry/action calibration or later task execution, not gross gripper timing.
Do not simply train ACT H20 longer; this config overfits validation quickly.
Next inspect videos and run a smaller/regularized or task-5-focused ACT diagnostic before scaling ACT or adding memory.
```
