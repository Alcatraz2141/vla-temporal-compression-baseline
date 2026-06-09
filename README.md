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

Latest RunPod rollout-facing state as of 2026-06-09:

```text
Confirmed per-task protocol:
  1. train phase-conditioned ACT single-task
  2. run split-aware rollouts
  3. train event-gated ACT from that task's phase ACT checkpoint
  4. run the same rollouts
  5. compare offline metrics and online success

Task 5 confirmation:
  phase ACT train20 / val5 / test5:        15/20 total, 8/10 held-out
  event-gated ACT train20 / val5 / test5:  29/30 total, 9/10 held-out
  summary: results/task5_event_memory_confirmation_20260608.md

Task 2 result:
  phase ACT train10 / val5 / test5:        15/20 total, 6/10 held-out
  phase ACT continued20:                   12/20 total, 6/10 held-out
  age-gated ACT continued20:                5/20 total, 2/10 held-out
  event-gated ACT train10 / val5 / test5:  19/20 total, 9/10 held-out
  summary: results/task2_event_memory_comparison_20260608.md
  control audit: results/task2_final_control_audit_20260608.md

Current interpretation:
  event-gated ACT has positive evidence on task 5 and task 2.
  task 2 is not explained by longer phase training or age/recency memory.
  corrected held-out offline eval does not explain the rollout gap.
  task 2 test-only is tied with original phase ACT at 4/5, so do not overclaim every split.

Kitchen4 from-scratch probe:
  task id: 3
  task: KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
  phase ACT train10 / val / test5:        3/10, 1/4, 2/5 = 6/19 total, 3/9 held-out
  event-gated ACT train10 / val / test5:  7/10, 4/4, 4/5 = 15/19 total, 8/9 held-out
  age-gated ACT train10 / val / test5:    3/10, 1/4, 1/5 = 5/19 total, 2/9 held-out
  offline continuous_mse:
    phase ACT:       0.05191242003440857
    event-gated ACT: 0.04404234265983105
    age-gated ACT:   0.047048382997512815
  summary: results/kitchen4_fromscratch_event_memory_20260609.md
  artifact backup:
    /workspace/run_backups/vla_run_artifacts_20260609_113957.tar.gz
    https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/cf4a2a58dce40859cf701b91da349781b25c44d6
  latest backup after age-gated control:
    /workspace/run_backups/vla_run_artifacts_20260609_180542.tar.gz
    https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/a5c98d68099f2f050941034b3737b21cd3ed5875

Current interpretation:
  event-gated ACT has positive evidence on task 5, task 2, and now task 3/kitchen4.
  kitchen4 is especially useful because the phase and event-gated models were both trained from scratch.
  kitchen4 age-gated from-scratch improves offline MSE over phase but does not reproduce the rollout gain.
  next, repeat the from-scratch protocol on one more task or run larger/multi-seed confirmation.
```

Diagnostic task-2 videos were generated after the measured rollouts:

```text
phase videos:
results/rollout_videos_phase_act_task2_diagnostics/act_chunked_corrected_h20_task2_phase_conditioned/

event-gated videos:
results/rollout_videos_event_gated_act_task2_diagnostics/event_gated_act_h20_task2_phase_memory/

episodes:
6, 7, 20, 29, 40, 41
```

Use those videos for qualitative inspection only. Some diagnostic rerun outcomes changed, which indicates rollout nondeterminism or context sensitivity; the measured rollout tables remain the primary results.

Previous RunPod rollout-facing state as of 2026-06-04:

```text
Current best task-5 controller:
  phase + event-gated memory ACT
  config: configs/libero_long_event_gated_act_h20_task5_phase_memory.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/event_gated_act_h20_task5_phase_memory/best.pt
  best epoch: 72
  offline continuous_mse: 0.013256419223546981
  rollout protocol: task-5 train10 / val5 / test5
  result: 17/20 total

Frozen controller baseline:
  phase-conditioned ACT
  config: configs/libero_long_act_chunked_corrected_h20_task5_phase_conditioned.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_phase_conditioned/best.pt
  rollout protocol: task-5 train10 / val5 / test5
  result: 10/20 total

Object-signal diagnostic:
  config: configs/libero_long_act_chunked_corrected_h20_task5_object_signals.yaml
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_object_signals/best.pt
  offline metrics improved
  rollout result: 10/20 total
  conclusion: do not include object-signal conditioning in the main comparison

Next controlled comparison:
  larger confirmation rollout for event-gated ACT
  task-5 train20 / val10 / test10

Decision rule:
  event-gated ACT already cleared the >=13/20 memory-helping threshold.
  next step is confirmation, not another model change.
```

Current artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260608_185154.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/7975ed9feed8299e08c88c8c30aaa81ecca01907
```

Previous artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260608_132450.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/3ef40e83472fc207cac83303bfa969dda647995f
```

Earlier artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260604_124932.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/83ba81224f5b6918ab4ffb55c245e8dc86c45ef4
```

Keep the rollout trace instrumentation active. Inspect failed event-gated ACT episodes before another model change:

```text
train ep8
train ep10
val ep45
```

Latest local diagnostic update as of 2026-06-02:

```text
The June 1 RunPod artifact was downloaded from:
runpod_20260601/vla_run_artifacts_20260601_132848.tar.gz

Extracted locally:
checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20/best.pt
checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt

Per-task validation diagnostics for full_transition20:
results/per_task_transition_diagnostics_transition20_val.csv

Summary:
- mean task continuous_mse:        0.000862
- mean task continuous_mae:        0.017611
- mean task gripper accuracy:      0.966178
- overall transition accuracy:     101/175 = 0.577143
- overall near-transition accuracy: 943/1216 = 0.775493

Task-5 checkpoint comparison:
- full_transition20 transition hits: 8/15
- task5_overfit transition hits:     12/15

Interpretation:
- Average action prediction is strong, but exact gripper transition timing is still too weak.
- Overall gripper accuracy is misleading because most frames are non-transition frames.
- The baseline must learn reliable grasp/release timing before event-memory results are interpretable.
```

Current recommended next work:

```text
1. Strengthen full-dataset task-balanced plus transition-balanced sampling.
2. Treat per-task transition accuracy as a primary gate.
3. If transition accuracy remains weak, add stronger task conditioning.
4. Train event-gated memory only after sliding-window uses the same improved protocol.
5. If improved sampling/conditioning still gives 0% rollout, pivot to ACT/action chunking.
```

Latest RunPod corrected-H1 rollout update as of 2026-06-01:

```text
GPU: RTX 4090, 24 GB VRAM

Main finding:
- Corrected-H1 rollout can succeed in LIBERO sim.
- Earlier 0% rollouts were not just a simulator wiring failure.
- The immediate issue was sparse gripper transition behavior and closed-loop brittleness.

Task-5 transition-aware overfit:
- checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
- best epoch: 20
- best val_loss: 0.0019988442626804096
- eval continuous_mse: 0.004982923693526134
- eval continuous_mae: 0.04651936175797483
- eval gripper_sign_accuracy: 0.9999000800313423

Rollouts:
- task-5 train split: 5/5
- task-5 val split: 2/5
- task-5 test split: 5/5

Useful videos:
- results/rollout_videos_sliding_window_corrected_h1_task5_overfit_train_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
- results/rollout_videos_sliding_window_corrected_h1_task5_overfit_test_task5/sliding_window_corrected_h1_task5_overfit/seed42_task05_episode6_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

Next run:
- full LIBERO corrected-H1 sliding-window, 20 epochs, transition sampling/loss enabled
- then eval, diagnostics, and task 0/2/5 rollouts
- then train corrected-H1 event-gated memory with the same 20-epoch protocol
```

Latest corrected-H1 local validation as of 2026-05-31:

```text
Status: GO for a bounded RunPod run, not a full blind sweep.

Implemented since the 2026-05-25 runs:
- corrected H=1 configs for sliding-window and event-gated memory
- action normalization from train-split stats
- binary gripper loss
- ImageNet image normalization
- unified-loader augmentation
- language conditioning for event-gated memory
- masked event-gate deltas
- split-aware rollout init selection
- rollout-aligned offline diagnostics

Local/Windows 4 GB VRAM validation:
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- PyTorch: 2.5.1+cu121
- CUDA available: yes
- LIBERO inspect and smoke test: passed
- tiny corrected sliding-window run decreased val_loss from 0.757886 to 0.745255
- 2 x 50-step check decreased val_loss from 0.750228 to 0.744029

Tiny-checkpoint diagnostics:
- continuous_mse:        0.7920950475
- continuous_mae:        0.5755884461
- first_action_mse:      0.8214608968
- position_mse:          0.8370953549
- rotation_mse:          0.7470947452
- gripper_sign_accuracy: 0.555

Interpretation:
- The corrected training/eval path is wired and learning.
- The tiny model is not rollout-ready.
- Next spend should be bounded RunPod training, not a full 50-epoch run by default.
```

Latest authoritative state as of 2026-05-25:

```text
sliding_window:
  best checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
  best epoch: 18
  best val_mse: 0.008474992022716574
  offline eval: MSE 0.059324943327478, MAE 0.2940476749624525
  task-5 rollout: 0/1

event_gated_memory:
  best checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
  best epoch: 46
  best val_mse: 0.008947615628130734
  offline eval: MSE 0.06263986602425575, MAE 0.2735399380326271
  task-5 rollout: 0/1
  video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory:
  best checkpoint: checkpoints/libero_long/age_gated_memory/best.pt
  best epoch: 31
  best val_mse: 0.010890026518609375
  last checkpoint: epoch 50, val_mse 0.014233005978167057
  offline eval: MSE 0.0762301841750741, MAE 0.25906969606876373
  task-5 rollout: 0/1
  video: results/rollout_videos_age_gated_memory_50ep/age_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

event_gated_concat_query:
  best checkpoint: checkpoints/libero_long/event_gated_concat_query/best.pt
  best epoch: 35
  best val_mse: 0.009582  # training log precision
  last checkpoint: epoch 50, val_mse 0.013192
  offline eval: MSE 0.06707473052665591, MAE 0.2805633209645748
  task-5 rollout: 0/1
  video: results/rollout_videos_concat_query_50ep/event_gated_concat_query/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
```

Restore sequence:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

uv run hf download Alcatraz1412/vla-run-backups   --repo-type dataset   --local-dir /workspace/run_backups

LATEST_BACKUP="$(ls -t /workspace/run_backups/vla_run_artifacts_*.tar.gz | head -1)"
tar -xzf "$LATEST_BACKUP" -C /root/vla-temporal-compression-baseline

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets   --repo-type dataset   --local-dir data/libero_long   --include "libero_10/*.hdf5"   --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
bash libero_rollout_env/bootstrap.sh
```

Next immediate work:

```text
Run a bounded corrected-H1 RunPod training gate:
1. Train configs/libero_long_sliding_window_corrected_h1.yaml for 2000-5000 steps.
2. Evaluate with evaluation/eval.py and evaluation/offline_diagnostics.py.
3. Continue only if val_loss drops clearly below the local 0.744 baseline and gripper_sign_accuracy improves beyond 0.555.
4. Then run configs/libero_long_event_gated_corrected_h1.yaml for the same step budget.
5. Run task-5 training-init rollout only after offline diagnostics are sane.

The older four-model single-seed offline table and task-5 diagnostic rollouts are complete, but they used pre-corrected H=4-style objectives/checkpoints and should not be treated as directly comparable to the new corrected-H1 runs.
Rollouts on RTX PRO 4500 Blackwell pods require --device cpu in the isolated rollout environment.
```

Before stopping a non-persistent pod:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
uv run huggingface-cli upload Alcatraz1412/vla-run-backups /workspace/run_backups --repo-type dataset
```

## Current Run: ACT H20 Corrected LIBERO

As of 2026-06-02, the corrected H1 task-balanced sliding-window baseline is stable offline but still fails train-init rollouts on tasks 0/2/5. The active next baseline is ACT/action chunking, not more H1 epochs.

```bash
uv run python train.py --config configs/libero_long_act_chunked_corrected_h20.yaml
```

Log:

```text
logs/act_chunked_corrected_h20_task_balanced_transition20_20260602.log
```

After training:

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
task 5 rollout with temporal ensembling: 0/3
```

Latest task-5 architecture diagnostics as of 2026-06-03:

```text
ACT task-5 overfit:
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit/best.pt
  rollout: 1/3

ACT task-5 consistency40:
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt
  rollout: 1/3

ACT placement-weighted55:
  checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_placement_weighted55/best.pt
  continuous_mse: 0.018447628365457058
  rollout: 1/3

Diffusion task5 H20 small:
  checkpoint: checkpoints/libero_long_corrected_task5/diffusion_task5_h20_small/best.pt
  stopped epoch: 35
  best val denoising loss: 0.19332740310662852
  sampled-action continuous_mse: 0.4715414630909697
  rollout: not run
```

Current interpretation:

```text
Placement weighting improves offline ACT error but does not improve closed-loop success.
Small diffusion fits on 24 GB VRAM and trains, but sampled actions are still much worse than ACT.
The next useful architecture direction is phase-conditioned ACT or a placement/refinement head.
Do not spend rollout time on the current diffusion checkpoint unless sampled-action eval improves substantially.
```

Previous artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260603_161004.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/d57c1138d2700872df6451f2f632b1db3db11a37
```

Trace comparison shows ACT improved the task-5 grasp timing/pose error versus sliding-window, but not enough for success:

```text
episode 0: 11 steps late, 0.0216 m from expert grasp pose
episode 1: 8 steps early, 0.0366 m from expert grasp pose
episode 2: 9 steps late, 0.0445 m from expert grasp pose
```

Next step: inspect ACT videos and run a smaller/regularized or task-5-focused ACT diagnostic. Do not rerun the same ACT config for more epochs.

## Current Run: ACT Task-5 Overfit Diagnostic

The smaller, task-5-only ACT diagnostic has now completed and is the active baseline path.

Config:

```text
configs/libero_long_act_chunked_corrected_h20_task5_overfit.yaml
```

Artifacts:

```text
log: logs/act_chunked_corrected_h20_task5_overfit_20260603_103917.log
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit/best.pt
eval row: results/baselines_corrected_task5.csv
rollout csv: results/libero_rollouts_act_chunked_h20_task5_overfit.csv
rollout trace: results/rollout_trace_act_chunked_h20_task5_overfit_train3.csv
```

Result summary:

```text
epoch 20 val_loss: 0.015803
continuous_mse: 0.029388979223370554
continuous_mae: 0.12436646746397019
gripper_sign_accuracy: 0.9982025062561035
task-5 train-init rollout: 1/3
```

Interpretation:

```text
This is the first ACT checkpoint with nonzero closed-loop task-5 success.
ACT is now validated enough to keep as the lead architecture.
The remaining failures are mostly post-grasp carry/place consistency, not gross approach or grasp discovery.
```

Immediate next run:

```bash
uv run python train.py --config configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
```

That continuation resumes from the epoch-20 task-5 checkpoint and pushes for 3/3 consistency before scaling back to multitask ACT.

## Current Result: ACT Placement Diagnostics

The epoch-40 task-5 ACT continuation did not improve rollout consistency:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
best checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt
continuous_mse: 0.027569980311393738
continuous_mae: 0.12052609633207322
gripper_sign_accuracy: 0.9987725071907043
task-5 train-init rollout: 1/3
```

The state-action diagnostic disabled vision and used only robot proprio plus action history:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_state_action.yaml
best checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_state_action/best.pt
continuous_mse: 0.11776154580116271
continuous_mae: 0.24411540160179138
gripper_sign_accuracy: 0.9767350002288818
task-5 train-init rollout: 0/3
```

This says vision is still needed; proprio/action history alone does not solve task 5.

The rollout script now has an expert-prefix handoff diagnostic:

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

Handoff results:

```text
normal ACT task-5 consistency40: 1/3
expert prefix 90:  1/3
expert prefix 130: 2/3
expert prefix 160: 1/3
```

Current interpretation:

```text
The bottleneck is placement/caddy insertion and recovery, not early gross approach alone.
The next useful experiment should target the placement phase directly: phase-conditioned ACT, placement-window oversampling/loss, or a separate placement/refinement policy.
Do not move to event memory or generic longer ACT training until the task-5 placement controller is made consistent.
```
