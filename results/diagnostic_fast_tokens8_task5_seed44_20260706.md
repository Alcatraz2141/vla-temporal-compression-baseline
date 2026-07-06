# Task 5 Fast-Token Event-vs-Age Diagnostic, Seed 44

Date: 2026-07-06

Task:

```text
task id for rollout: 5
task filter: STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
```

Protocol:

```text
seed: 44
samples_per_epoch: 5000
epochs: 30
batch_size: 32
chunk_size: 4
max_memory_tokens: 8
older context frames: 32
cache_max_episodes: 128
num_workers: 8
prefetch_factor: 2
rollout seed: 42
rollout max_steps: 300
rollout temporal_ensemble: true
rollout splits: val5 / test5 split-aware task-5 episodes
```

## Training

Event-gated:

```text
config: configs/diagnostic_event_gated_act_task5_seed44_fast_tokens8.yaml
log: logs/diagnostic_event_gated_task5_seed44_fast_tokens8_20260706.log
completed epoch: 30
best epoch: 29
best val_loss: 0.051597
epoch-30 val_loss: 0.055997
checkpoint: checkpoints/diagnostic_event_gated_task5_seed44_fast_tokens8/diagnostic_event_gated_act_h20_task5_fast_tokens8_seed44/best.pt
```

Age-gated:

```text
config: configs/diagnostic_age_gated_act_task5_seed44_fast_tokens8.yaml
log: logs/diagnostic_age_gated_task5_seed44_fast_tokens8_20260706.log
completed epoch: 30
best epoch: 28
best val_loss: 0.052710
epoch-30 val_loss: 0.060673
checkpoint: checkpoints/diagnostic_age_gated_task5_seed44_fast_tokens8/diagnostic_age_gated_act_h20_task5_fast_tokens8_seed44/best.pt
```

## Offline Eval

Event-gated epoch-29 best:

```text
continuous_mse: 0.06712389662861824
continuous_mae: 0.1846827745437622
gripper_sign_accuracy: 0.9856250047683716
pred_temporal_smoothness: 0.04199544116854668
```

Age-gated epoch-28 best:

```text
continuous_mse: 0.06931269615888595
continuous_mae: 0.18851559460163117
gripper_sign_accuracy: 0.9868749856948853
pred_temporal_smoothness: 0.06085030883550644
```

## Held-Out Rollout

Event-gated epoch-29 best:

```text
val:  3/5
test: 2/5
held-out: 5/10
val failures:  [27, 36]
test failures: [14, 21, 39]
```

Age-gated epoch-28 best:

```text
val:  3/5
test: 5/5
held-out: 8/10
val failures:  [27, 45]
test failures: []
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task5_seed44_fast_tokens8_val5_epoch29_best_20260706.csv
results/diagnostic_rollouts_event_gated_task5_seed44_fast_tokens8_test5_epoch29_best_20260706.csv
results/diagnostic_rollouts_age_gated_task5_seed44_fast_tokens8_val5_epoch28_best_20260706.csv
results/diagnostic_rollouts_age_gated_task5_seed44_fast_tokens8_test5_epoch28_best_20260706.csv
```

Trace files:

```text
results/diagnostic_trace_event_gated_task5_seed44_fast_tokens8_val5_epoch29_best_20260706.csv
results/diagnostic_trace_event_gated_task5_seed44_fast_tokens8_test5_epoch29_best_20260706.csv
results/diagnostic_trace_age_gated_task5_seed44_fast_tokens8_val5_epoch28_best_20260706.csv
results/diagnostic_trace_age_gated_task5_seed44_fast_tokens8_test5_epoch28_best_20260706.csv
```

## Readout

Task 5 is rollout-competent under the cheap 30-epoch, 5000-samples-per-epoch, 8-token protocol,
unlike task 8. Age-gated and event-gated have very similar training validation and offline action
prediction, but age-gated is much stronger online on the held-out test subset.

```text
event-gated: continuous_mse 0.067124, held-out 5/10
age-gated:   continuous_mse 0.069313, held-out 8/10
```

The important comparison is that age-gated is slightly worse offline but substantially better in
closed-loop rollout. This matches the broader warning from task 2: offline continuous MSE is not a
reliable method selector, and the current event-gate design is not consistently outperforming a
simple age/recency gate.
