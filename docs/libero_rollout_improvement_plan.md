# LIBERO Rollout Improvement Plan

Date: 2026-05-30

## 2026-06-28 Seed-44 Event-Gated Restart And Validation Decoupling

The task-2 seed-44 event-gated ACT run was restarted from scratch after fixing training-time
validation overhead.

Implementation:

```text
datasets/data_loader.py:
  build_dataloader now uses shuffle=True/False as the train/eval mode signal.

datasets/episode_loader.py and datasets/episode_dataset.py:
  datasets now accept an explicit training flag, so validation can read train-split records
  deterministically without stochastic train sampling.

training/train.py:
  epoch summaries now print train_seconds and val_seconds.
```

Effect:

```text
before:
  val_split=train caused validation to run 20k stochastic train-mode samples.
  validation cost was roughly 9-10 minutes per epoch.

after:
  validation over the task-2 train split is 148 deterministic samples / 5 batches.
  validation cost is about 3-4 seconds per epoch.
  event-gated seed-44 epoch time is about 10.2 minutes.
```

Seed-44 outcome:

```text
checkpoint root: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory
stopped epoch: 50
epoch-50 last.pt offline continuous_mse: 0.04505929201841354
epoch-50 last.pt offline continuous_mae: 0.14305126070976257
epoch-50 rollout train30 / val5 / test5: 17/30, 3/5, 4/5 = 24/40
epoch-50 held-out val+test: 7/10

epoch-46 best.pt rollout train30 / val5 / test5: 17/30, 2/5, 2/5 = 21/40
artifact backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/40def1523780664f7d84a1402c8294be0b8fdffa
```

Seed-44 epoch-60 continuation audit:

```text
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
completed epoch: 60
offline continuous_mse: 0.04118259623646736
offline continuous_mae: 0.13802042752504348
rollout train30 / val5 / test5: 16/30, 1/5, 1/5 = 18/40
held-out val+test: 2/10
summary: results/paper_event_gated_task2_seed44_epoch60_20260629.md
```

Conclusion:

```text
The speed fix should be kept. It changes training-time validation and best.pt selection, not
policy training updates or rollout logic.

For seed-44 event-gated reporting, use epoch-50 last.pt rather than the decoupled-validation
best.pt or epoch-60 last.pt. The best.pt checkpoint was worse on held-out rollout, and epoch 60
improved offline prediction while substantially worsening rollout.
```

Seed-187 event-gated epoch-50 stop and rollout:

```text
config: configs/paper_event_gated_act_task2_seed187.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed187/event_gated_act_h20_task2_phase_memory/last.pt
stopped after completed epoch: 50
offline continuous_mse: 0.04214628413319588
offline continuous_mae: 0.13997110724449158
rollout train30 / val5 / test5: 17/30, 2/5, 1/5 = 20/40
held-out val+test: 3/10
summary: results/paper_event_gated_task2_seed187_epoch50_20260629.md
```

Compared with matched phase ACT seed 187 at `25/40` total and `5/10` held-out, seed-187
event-gated is weaker in rollout. The task-2 from-scratch event-gated paper-seed result should
be framed as mixed/negative relative to phase ACT, despite reasonable offline metrics.

Seed-187 cold-start phase ACT audit:

```text
config: configs/paper_phase_act_task2_seed187_cold_start_resume.yaml
checkpoint: checkpoints/paper_phase_act_task2_seed187_cold_start/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
completed epoch: 60
best checkpoint by training validation: epoch 58
offline continuous_mse: 0.03023523513815905
offline continuous_mae: 0.12312478846625279
rollout train30 / val5 / test5: 25/30, 4/5, 4/5 = 33/40
held-out val+test: 8/10
summary: results/paper_phase_act_task2_seed187_cold_start_20260630.md
```

This makes the seed-187 comparison more negative for event-gated memory than the older matched
phase reference: cold-start phase ACT is better offline and much better online. Continue
event-gated seed 187 to epoch 60 as an audit, not because epoch 50 supports the memory claim.

Event-gated seed-187 epoch-51-to-60 audit start:

```text
resume config: configs/paper_event_gated_act_task2_seed187_resume.yaml
restored epoch-50 checkpoint from: Alcatraz1412/vla-run-backups/vla_run_artifacts_20260629_180122.tar.gz
log: logs/paper_event_gated_task2_seed187_resume_51_60_20260630.log
epoch 51: train_loss 0.027196, val_loss 0.069595
epoch 52: train_loss 0.026709, val_loss 0.056911
status after stability check: stable, last.pt at epoch 52, continuing toward epoch 60
```

Event-gated seed-187 epoch-60 audit result:

```text
config: configs/paper_event_gated_act_task2_seed187_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed187/event_gated_act_h20_task2_phase_memory/last.pt
completed epoch: 60
best checkpoint by training validation: epoch 56
epoch-60 offline continuous_mse: 0.03697693571448326
epoch-56 best.pt offline continuous_mse: 0.036199239641427995
rollout train30 / val5 / test5: 13/30, 2/5, 2/5 = 17/40
held-out val+test: 4/10
summary: results/paper_event_gated_task2_seed187_epoch60_20260630.md
```

Epoch 60 improves offline action prediction but worsens total rollout relative to epoch 50.
The task-2 event-gated result remains negative relative to cold-start phase ACT, especially seed
187 where phase ACT reaches 33/40 total and 8/10 held-out.

Age-gated ACT seed-43 epoch-50 audit:

```text
config: configs/paper_age_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed43/age_gated_act_h20_task2_phase_memory_seed43/last.pt
stopped after completed epoch: 50
offline continuous_mse: 0.038894045352935794
offline continuous_mae: 0.13535064458847046
rollout train30 / val5 / test5: 20/30, 4/5, 3/5 = 27/40
held-out val+test: 7/10
summary: results/paper_age_gated_task2_seed43_epoch50_20260630.md
```

This age-gated control outperforms event-gated seed 43 online while underperforming it offline.
That is a direct warning that the current event gate is not selecting memory better than simple
age weighting on task 2, and that offline MSE cannot be used as the primary method selector.

Age-gated seed-43 epoch-60 audit:

```text
config: configs/paper_age_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed43/age_gated_act_h20_task2_phase_memory_seed43/last.pt
completed epoch: 60
offline continuous_mse: 0.036862388253211975
offline continuous_mae: 0.13023996502161025
rollout train30 / val5 / test5: 21/30, 2/5, 3/5 = 26/40
held-out val+test: 5/10
```

Age-gated seed-44 epoch-58 audit:

```text
config: configs/paper_age_gated_act_task2_seed44.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed44/age_gated_act_h20_task2_phase_memory_seed44/last.pt
stopped after completed epoch: 58
best checkpoint by training validation: epoch 56
epoch-58 offline continuous_mse: 0.05122858919203281
best.pt offline continuous_mse: 0.03577113375067711
rollout train30 / val5 / test5: 21/30, 3/5, 1/5 = 25/40
held-out val+test: 4/10
summary: results/paper_age_gated_task2_seed44_epoch58_20260701.md
artifact backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/cffb17c7e30e71e53d06ed368511abcf627601e2
```

Seed-44 age-gated total rollout is competitive with seed-43, but held-out rollout is weaker.
The validation-selected best.pt is much better offline than epoch-58 last.pt, so it should be
rolled out before finalizing seed-44 reporting.

Age-gated seed-187 partial run:

```text
config: configs/paper_age_gated_act_task2_seed187.yaml
resume config: configs/paper_age_gated_act_task2_seed187_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed187/age_gated_act_h20_task2_phase_memory_seed187/last.pt
stopped after completed epoch: 27
best checkpoint by training validation: epoch 24
best val_mse: 0.05992962270975113
epoch 27 val_mse: 0.061624
summary: results/paper_age_gated_task2_seed187_epoch27_20260701.md
artifact backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/dfc6e281436abc176783f2f59c57b4ce87e6d95d
```

The log contains one epoch-28 step because the stop command landed after epoch 28 started. The
saved checkpoint is epoch 27; resume from `last.pt` with the seed-187 resume config.

## 2026-06-27 Seed-44 Stop And Frozen-Vision Speed Diagnostic

The next matched from-scratch task-2 event-gated seed was started and then stopped on request.

```text
seed: 44
config: configs/paper_event_gated_act_task2_seed44.yaml
resume config: configs/paper_event_gated_act_task2_seed44_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
last completed epoch: 11
best val_mse: 0.06793733193278313
```

Resume with the seed-44 resume config; the unfinished epoch-12 work was discarded.

A frozen ResNet18 diagnostic was run to test whether event-gated ACT is bottlenecked by visual
backprop.

```text
config: configs/diagnostic_event_gated_act_task2_seed44_freeze_vision.yaml
model change: ACTChunkedBaseline supports model.freeze_vision=true
last completed epoch: 7
best val_mse: 0.14847677819728852
```

Result:

```text
Freezing ResNet18 reduced VRAM, but did not materially reduce wall-clock epoch time.
Frozen epochs remained about 16.5-17 minutes, similar to the unfrozen seed-44 run.
The frozen model also learned more slowly early in training.
```

Diagnosis:

```text
The likely bottleneck is the event-gated ACT older-context image pipeline and validation path:
K_recent 8 + older_obs 64 means 72 images/sample, or about 2304 images/batch at batch size 32.
GPU utilization is bursty, with many 0% samples between compute bursts. This points to data
loading / HDF5 / CPU preprocessing / validation overhead rather than ResNet backward compute.
```

Next speed work should test larger task-2 episode caching and explicit train-vs-validation timing
before changing memory token count or model architecture.

Artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260627_113647.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/513bfd80a278f3f22d0b874f70709bf36aecc147
```

## 2026-06-26 Task-2 From-Scratch Event-Gated Seed 43 Completed

The task-2 event-gated ACT seed-43 from-scratch run resumed from epoch 21 and completed to
epoch 60.

```text
config: configs/paper_event_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory/best.pt
best epoch: 58
best val_mse: 0.020923468711972235
offline continuous_mse: 0.01848969299942255
offline continuous_mae: 0.09646275240182876
gripper_sign_accuracy: 0.9892899991989136
```

Unique split-aware rollout:

```text
train30 / val5 / test5: 18/30, 3/5, 3/5 = 24/40
held-out val+test: 6/10
```

Interpretation:

```text
This run is an offline-positive but rollout-negative result relative to seed-43 phase ACT.
On the comparable train10 / val5 / test5 subset, phase ACT seed 43 was 17/20 while
from-scratch event-gated seed 43 was 12/20.

The result reinforces closed-loop seed/training sensitivity and the weakness of offline MSE as
a checkpoint selector. Finish matched from-scratch event-gated seeds 44 and 187 before beginning
age-gated controls.
```

Artifacts:

```text
results/paper_event_gated_task2_seed43_20260626.md
results/paper_baselines_event_gated_task2_seed43.csv
results/paper_rollouts_event_gated_task2_seed43_{train30,val5,test5}.csv
results/paper_trace_event_gated_task2_seed43_{train30,val5,test5}.csv
backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/aee8f20fd7770fc071239ecd9ee75190d423e21b
```

## 2026-06-24 Task-2 Phase-ACT Seed 187 And Matched Event Plan

```text
seed 187 best epoch: 58
offline continuous_mse: 0.019253734470903873
rollout train30 / val5 / test5: 20/30, 1/5, 4/5 = 25/40
held-out val+test: 5/10
```

This seed reinforces closed-loop seed variance: offline metrics are strong, while held-out rollout
is weaker than seed 43. The matched event protocol is from-scratch 60-epoch training, ordered
seed 43, seed 44, then seed 187. Evaluate each with unique train30/val5/test5 rollouts.
Begin age-gated controls only after the event-gated multi-seed results are recorded.

Artifact backup:

```text
/workspace/run_backups/vla_run_artifacts_20260624_121312.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/bf0eecd96297788fabe223fbb6099b1735703945
```

Current execution state:

```text
event-gated seed 43 from scratch
completed/stopped epoch: 21
best val_mse: 0.04350930461883545
resume config: configs/paper_event_gated_act_task2_seed43_resume.yaml
```

Finish seed 43 to epoch 60 before starting from-scratch seeds 44 and 187.

Epoch-21 shutdown backup:

```text
/workspace/run_backups/vla_run_artifacts_20260624_185606.tar.gz
https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/02e537abdbdf6587d8f268517b35eb36290a32a3
```

## 2026-06-23 Task-2 Phase-ACT Seed Variance Check

The task-2 phase-ACT baseline was expanded with paper seed runs.

```text
task id: 2
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
protocol: train10 / val5 / test5, max steps 300
simulator seed: 42
```

Results:

```text
seed 42 original phase ACT:
  checkpoint: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
  best epoch: 29
  rollouts: train10 9/10, val5 2/5, test5 4/5 = 15/20

seed 43 phase ACT:
  checkpoint: checkpoints/paper_phase_act_task2_seed43/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
  best epoch: 57
  offline continuous_mse: 0.01966886719018221
  rollouts: train10 8/10, val5 4/5, test5 5/5 = 17/20

seed 44 phase ACT:
  checkpoint: checkpoints/paper_phase_act_task2_seed44/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
  best epoch: 58
  offline continuous_mse: 0.020462753280997278
  rollouts: train10 4/10, val5 3/5, test5 3/5 = 10/20
```

Interpretation:

```text
Task-2 phase ACT has high online variance across training seeds.
Seed 43 is a strong controller; seed 44 is not, despite similar offline metrics.
Offline continuous MSE is insufficient for selecting the best rollout checkpoint.
The next clean paper step is matched event-gated memory training/evaluation for seeds 43 and 44,
not another single-seed task search.
```

Current next work:

```text
1. Train event-gated ACT task-2 phase-memory for seed 43 from the seed-43 phase checkpoint.
2. Train event-gated ACT task-2 phase-memory for seed 44 from the seed-44 phase checkpoint.
3. Run the exact same task-2 train10 / val5 / test5 rollouts.
4. Compare mean and per-episode flips against phase ACT.
```

Artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260623_191057.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/b415453cb949fd7cd6ebb2ba8abdae9a2c0ed72b
```

## 2026-06-09 Kitchen4 From-Scratch Event-Memory Result

The event-memory comparison was repeated on a new task with both models trained from scratch.

```text
task id: 3
task: KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
phase config: configs/libero_long_act_chunked_h20_kitchen4_drawer_phase_fromscratch.yaml
event config: configs/libero_long_event_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch.yaml
age config: configs/libero_long_age_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch.yaml
epochs: 20 each
seed: 42
```

Offline metrics:

```text
phase ACT continuous_mse:       0.05191242003440857
event-gated continuous_mse:     0.04404234265983105
age-gated continuous_mse:       0.047048382997512815
phase ACT continuous_mae:       0.162862943983078
event-gated continuous_mae:     0.15004283199310303
age-gated continuous_mae:       0.15670564353466035
```

Rollouts:

```text
phase ACT:        train10 3/10, val 1/4, test5 2/5 = 6/19
event-gated ACT:  train10 7/10, val 4/4, test5 4/5 = 15/19
age-gated ACT:    train10 3/10, val 1/4, test5 1/5 = 5/19
held-out:
  phase ACT:       3/9
  event-gated ACT: 8/9
  age-gated ACT:   2/9
```

Matched flips:

```text
phase failure -> event success:
  train: [2, 3, 6, 9]
  val:   [1, 8, 20]
  test:  [15, 37]

phase success -> event failure:
  none

event success -> age failure:
  train: [2, 3, 4, 6, 9]
  val:   [1, 8, 20]
  test:  [10, 15, 37]
```

Interpretation:

```text
This is a strong positive result for event-gated ACT under a from-scratch protocol.
The age-gated control improves offline MSE over phase ACT but does not reproduce the event-gated rollout gain.
This supports event-aware memory selection over simple recency/age selection on this task.
Do not treat this as multi-seed proof yet; it is still one seed and a selected rollout set.
```

Artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260609_113957.tar.gz
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/cf4a2a58dce40859cf701b91da349781b25c44d6

latest local: /workspace/run_backups/vla_run_artifacts_20260609_180542.tar.gz
latest HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/a5c98d68099f2f050941034b3737b21cd3ed5875
```

## 2026-06-04 Event-Gated ACT Result

The controlled memory comparison is now positive.

```text
baseline: event_gated_act
config: configs/libero_long_event_gated_act_h20_task5_phase_memory.yaml
best checkpoint: checkpoints/libero_long_corrected_task5/event_gated_act_h20_task5_phase_memory/best.pt
best epoch: 72
best val_loss: 0.008460610160976649
offline continuous_mse: 0.013256419223546981
offline continuous_mae: 0.08408326748609543
gripper_sign_accuracy: 0.9983475049972534
```

Rollout comparison under the same task-5 train10 / val5 / test5 protocol:

```text
phase ACT:        4/10 train, 3/5 val, 3/5 test = 10/20
object-signals:   5/10 train, 3/5 val, 2/5 test = 10/20
event-gated ACT:  8/10 train, 4/5 val, 5/5 test = 17/20
```

Decision:

```text
The memory criterion was >= 13/20.
Event-gated ACT reached 17/20, so memory is helping under this controlled task-5 protocol.
Event-gated ACT is the current best task-5 controller, pending larger confirmation.
```

Do next:

```text
1. Inspect failed event-memory episodes: train ep8, train ep10, val ep45.
2. Run larger confirmation rollout: train20 / val10 / test10.
3. Do not add another model change until the larger rollout confirms or weakens this result.
```

## 2026-06-07 Multi-Task Track-A Result

After checkpoint/log inspection, the task-5 event-gated ACT result should be framed as a
warm-start result:

```text
Starting from the same phase ACT checkpoint, event-gated memory fine-tuning improved task-5
rollout from 10/20 to 17/20. Object-signal fine-tuning from that checkpoint stayed at 10/20.
```

Full LIBERO-10 phase-conditioned ACT was tested as a possible multi-task baseline:

```text
config: configs/libero_long_act_chunked_h20_multitask_phase_conditioned.yaml
checkpoint: checkpoints/libero_long_multitask_track_a/act_chunked_h20_multitask_phase_conditioned/last.pt
offline continuous_mse: 0.26775041222572327
offline continuous_mae: 0.34758521403585163
gripper_sign_accuracy: 0.9363839200564793
rollouts:
  train1 all tasks: 4/10
  train3 all tasks: 9/30
  val1 all tasks:   0/10
  test1 all tasks:  0/10
```

This is not a credible full-suite baseline yet because held-out rollout success is zero.

A three-task subset was also tested:

```text
tasks: 1, 2, 4
config: configs/libero_long_act_chunked_h20_subset124_phase_conditioned.yaml
checkpoint: checkpoints/libero_long_subset_track_a/act_chunked_h20_subset124_phase_conditioned/last.pt
offline continuous_mse: 0.36372560262680054
offline continuous_mae: 0.4136500895023346
gripper_sign_accuracy: 0.9315624952316284
rollouts:
  train5: 3/15
  val3:   1/9
  test3:  0/9
```

Interpretation:

```text
The selected subset is still too weak for a clean event-memory comparison.
The only held-out success came from task 2 on val; test stayed zero.
Do not spend on full multitask event-gated memory as a paper result until phase ACT has nonzero
held-out rollout on the chosen task set.
```

Recommended next options:

```text
1. Use task 2 alone as a minimal nonzero held-out diagnostic for phase ACT vs event memory.
2. Search for a different 2-3 task subset with nonzero val/test phase-ACT success.
3. Improve ACT generalization before adding memory.
```

Latest uploaded artifacts:

```text
full multitask / initial subset backup:
  /workspace/run_backups/vla_run_artifacts_20260607_155315.tar.gz
  https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/df50d8d23b8eb785437fa5f59b588561ba916969

subset eval / rollout backup:
  /workspace/run_backups/vla_run_artifacts_20260607_164317.tar.gz
  https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/2b15a797510ccde40aec6bcc605599c71dc32627
```

## 2026-06-04 Current Decision Point

The current rollout-facing controller baseline is phase-conditioned ACT on task 5.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_phase_conditioned.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_phase_conditioned/best.pt
best epoch: 58
continuous_mse: 0.019623747254908085
continuous_mae: 0.10296388152837753
gripper_sign_accuracy: 0.9973975031852722
task-5 rollout protocol: train10 / val5 / test5
task-5 rollout result: 4/10 train, 3/5 val, 3/5 test, 10/20 total
```

The derived object-signal ACT run is a negative result:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_object_signals.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_object_signals/best.pt
best epoch: 69
continuous_mse: 0.01562450560182333
continuous_mae: 0.09162709747552872
gripper_sign_accuracy: 0.9983525047302246
task-5 rollout result: 5/10 train, 3/5 val, 2/5 test, 10/20 total
```

Interpretation:

```text
Offline action prediction improved with secured/placement_ready embeddings.
Closed-loop success did not improve.
The current object-signal heuristic should not be part of the main comparison.
Keep the trace instrumentation, but drop object-signal conditioning for the next run.
```

Next controlled research comparison:

```text
phase ACT baseline
vs
phase + event-gated memory ACT
```

Use the exact same task-5 rollout protocol:

```text
train-init: 10 episodes
val:        5 episodes
test:       5 episodes
max steps:  300
same split files and rollout settings
```

Decision rule:

```text
event-memory phase ACT >= 13/20:
  memory is helping; continue.

event-memory phase ACT around 10/20:
  memory is not the current bottleneck; report phase ACT as controller baseline and stop this detour.

event-memory phase ACT below 10/20:
  memory integration is adding noise or undertraining.
```

Run cheap rollout-only gripper execution ablations before or alongside the memory training:

```text
current temporal ensemble
no gripper ensembling
first-action gripper only
hysteresis threshold
```

Do not run another object-signal model before a trace shows a concrete fix.

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

### 2026-06-03 Task-5 ACT Diagnostic Result

The task-5-focused ACT diagnostic was run with a smaller and more regularized model:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_overfit.yaml
log: logs/act_chunked_corrected_h20_task5_overfit_20260603_103917.log
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_overfit/best.pt
```

Outcome:

```text
epoch 20 val_loss: 0.015803
continuous_mse: 0.029388979223370554
continuous_mae: 0.12436646746397019
gripper_sign_accuracy: 0.9982025062561035
task-5 train-init rollout: 1/3
```

Trace summary:

```text
episode 0: +16 steps vs expert first positive gripper, 0.0453 m grasp-pose error, success
episode 1: -11 steps, 0.0358 m, fail
episode 2: +12 steps, 0.0396 m, fail
```

Meaning:

```text
ACT can now solve task 5 in closed loop, so the previous ACT 0/3 result was not a rollout-interface dead end.
The remaining variance is mostly late carry/place behavior rather than initial approach.
The next milestone is consistency: push task-5 ACT from 1/3 to 3/3 before scaling back to multitask ACT.
```

Next experiment:

```text
Resume the task-5 ACT run from epoch 20 to epoch 40 in a new run directory.
Use the epoch-20 last checkpoint as the resume source.
Keep the model/task setup fixed and treat this as a consistency continuation, not a fresh architecture sweep.
```

### 2026-06-03 Placement Diagnostic Update

The epoch-40 consistency continuation completed but stayed at `1/3` task-5 train-init success:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_consistency40.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_consistency40/best.pt
continuous_mse: 0.027569980311393738
continuous_mae: 0.12052609633207322
gripper_sign_accuracy: 0.9987725071907043
rollout: 1/3
```

The state-action/proprio-only diagnostic did worse:

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_state_action.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_state_action/best.pt
continuous_mse: 0.11776154580116271
continuous_mae: 0.24411540160179138
gripper_sign_accuracy: 0.9767350002288818
rollout: 0/3
```

Expert-prefix handoff was added to `evaluation/libero_rollout.py`:

```text
--expert-prefix-steps N
```

This replays the matching HDF5 demo actions for `N` simulator steps, appends them to online history, then hands off to the trained ACT policy.

Handoff results on task 5 train demos `[0, 1, 2]`:

```text
normal ACT task-5 consistency40: 1/3
expert prefix 90:  1/3
expert prefix 130: 2/3
expert prefix 160: 1/3
```

Interpretation:

```text
The current failure is placement/caddy insertion and recovery.
Expert early motion alone does not solve it.
Handoff around step 130 helps, so reaching the right placement approach state matters.
Late handoff is still fragile, so final contact/release also needs better modeling.
```

Next recommended experiment:

```text
Do not train event memory next.
Do not run generic longer ACT training next.
Use a placement-focused ACT change:
1. placement-window oversampling and/or higher placement loss as the fastest diagnostic, or
2. phase-conditioned ACT if a stable phase label can be inferred from demo timestep/state.
```

### 2026-06-03 Late Update: Placement Weighting And Diffusion

Placement-window weighted ACT was tested first because it was the smallest change.

```text
config: configs/libero_long_act_chunked_corrected_h20_task5_placement_weighted55.yaml
checkpoint: checkpoints/libero_long_corrected_task5/act_chunked_corrected_h20_task5_placement_weighted55/best.pt
best epoch: 51
continuous_mse: 0.018447628365457058
continuous_mae: 0.10044001704454422
gripper_sign_accuracy: 0.9980300048828125
task-5 train-init rollout: 1/3
```

Result:

```text
The offline metric improved, but closed-loop success did not improve.
The failed episodes showed worse grasp-pose errors than the success case.
Conclusion: placement loss weighting alone is not enough and can perturb earlier approach/grasp behavior.
```

A separate small `diffusion_policy` baseline was added and trained on task 5 H20.

```text
config: configs/libero_long_diffusion_task5_h20_small.yaml
continuation: configs/libero_long_diffusion_task5_h20_small_to50.yaml
checkpoint: checkpoints/libero_long_corrected_task5/diffusion_task5_h20_small/best.pt
stopped epoch: 35
best val denoising loss: 0.19332740310662852
35-epoch sampled-action continuous_mse: 0.4715414630909697
35-epoch sampled-action continuous_mae: 0.44337508891718075
35-epoch gripper_sign_accuracy: 0.9636681321710824
```

Result:

```text
The small diffusion model fits on 24 GB VRAM and trains stably.
It is still not rollout-ready: sampled actions are far worse than ACT, even after deterministic sampling and 35 epochs.
Do not run simulator rollout for this checkpoint unless offline sampled-action metrics improve substantially.
```

Current direction:

```text
The best ACT policy remains a 1/3 closed-loop task-5 policy.
The failure mode still points to placement/recovery and action-distribution quality.
Next architecture work should be phase-conditioned ACT or a placement/refinement head, not another generic loss-weighting pass.
```

### 2026-06-08 Update: Event-Memory Is Now Positive On Two Per-Task ACT Runs

This plan has been superseded by the phase-conditioned ACT and event-gated ACT results.
The useful comparison is now per-task:

```text
phase-conditioned ACT single-task
vs
event-gated ACT warm-started from that task's phase ACT checkpoint
```

Task 5 confirmation:

```text
phase ACT:
  train20: 15/20
  val5:     4/5
  test5:    4/5
  total:   23/30
  held-out val+test: 8/10

event-gated ACT:
  train20: 20/20
  val5:     4/5
  test5:    5/5
  total:   29/30
  held-out val+test: 9/10
```

Task 2 result:

```text
task:
  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it

phase ACT:
  continuous_mse: 0.033583113986253736
  continuous_mae: 0.13209211629629136
  train10 / val5 / test5: 9/10, 2/5, 4/5 = 15/20
  held-out val+test: 6/10

event-gated ACT:
  continuous_mse: 0.022640780751407148
  continuous_mae: 0.10798589040040969
  train10 / val5 / test5: 10/10, 5/5, 4/5 = 19/20
  held-out val+test: 9/10

phase ACT continued20:
  train10 / val5 / test5: 6/10, 3/5, 3/5 = 12/20
  held-out val+test: 6/10

age-gated ACT continued20:
  train10 / val5 / test5: 3/10, 2/5, 0/5 = 5/20
  held-out val+test: 2/10
```

Interpretation:

```text
The old concern that memory should not be tried before the controller was stable is resolved
for these per-task ACT runs. Event-gated memory has now improved task 5 and task 2.

For task 2, the event-gated result is not explained by longer phase-ACT training or by
generic age/recency memory. Corrected held-out offline eval also does not explain the
closed-loop rollout gap.

Task 2 test-only is tied at 4/5, so the claim should stay precise:
event memory improves offline prediction and aggregate split-aware rollout success, but
additional tasks / seeds are needed before broad generalization claims.
```

Current best next steps:

```text
1. Repeat the same protocol on another LIBERO-Long task.
2. Run larger or multi-seed confirmation on tasks 2 and 5.
3. Then add ACT-memory ablations:
   - age gate
   - ACT query/mechanism cleanup; current memory.query_type is ignored by event_gated_act
   - memory-token count
   - phase-aware older memory by passing older_phase_ids into event-memory context encoding
```

Relevant summaries:

```text
results/task5_event_memory_confirmation_20260608.md
results/task2_event_memory_comparison_20260608.md
results/task2_phase_continued_control_20260608.md
results/task2_age_gated_control_20260608.md
results/task2_final_control_audit_20260608.md
```
