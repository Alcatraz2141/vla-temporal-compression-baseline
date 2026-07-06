# Task 3 Fast-Token Event-vs-Age Diagnostic, Seed 44

Date: 2026-07-06

Task:

```text
task id for rollout: 3
task filter: KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
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
rollout splits: val / test split-aware task-3 episodes
```

The split-aware val file contained four task-3 episodes, so held-out total is 9 episodes.

## Training

Event-gated:

```text
config: configs/diagnostic_event_gated_act_task3_seed44_fast_tokens8.yaml
log: logs/diagnostic_event_gated_task3_seed44_fast_tokens8_20260706.log
completed epoch: 30
best epoch: 30
best val_loss: 0.135667
checkpoint: checkpoints/diagnostic_event_gated_task3_seed44_fast_tokens8/diagnostic_event_gated_act_h20_task3_fast_tokens8_seed44/best.pt
```

Age-gated:

```text
config: configs/diagnostic_age_gated_act_task3_seed44_fast_tokens8.yaml
log: logs/diagnostic_age_gated_task3_seed44_fast_tokens8_20260706.log
last saved epoch: 28
best epoch: 24
best val_loss: 0.13137015948692957
checkpoint: checkpoints/diagnostic_age_gated_task3_seed44_fast_tokens8/diagnostic_age_gated_act_h20_task3_fast_tokens8_seed44/best.pt
```

The age-gated process exited after logging epoch-29 step 50. The saved `last.pt` is epoch 28;
partial epoch-29 work is not used.

## Offline Eval

Event-gated epoch-30 best:

```text
continuous_mse: 0.17221714556217194
continuous_mae: 0.27457812428474426
gripper_sign_accuracy: 0.9492187400658926
pred_temporal_smoothness: 0.06464940433700879
```

Age-gated epoch-24 best:

```text
continuous_mse: 0.16428105533123016
continuous_mae: 0.28328806658585864
gripper_sign_accuracy: 0.9486979146798452
pred_temporal_smoothness: 0.0677892913420995
```

## Held-Out Rollout

Event-gated epoch-30 best:

```text
val:  1/4
test: 0/5
held-out: 1/9
val successes:  [20]
test successes: []
```

Age-gated epoch-24 best:

```text
val:  2/4
test: 2/5
held-out: 4/9
val successes:  [1, 39]
test successes: [10, 35]
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task3_seed44_fast_tokens8_val5_epoch30_best_20260706.csv
results/diagnostic_rollouts_event_gated_task3_seed44_fast_tokens8_test5_epoch30_best_20260706.csv
results/diagnostic_rollouts_age_gated_task3_seed44_fast_tokens8_val5_epoch24_best_20260706.csv
results/diagnostic_rollouts_age_gated_task3_seed44_fast_tokens8_test5_epoch24_best_20260706.csv
```

Trace files:

```text
results/diagnostic_trace_event_gated_task3_seed44_fast_tokens8_val5_epoch30_best_20260706.csv
results/diagnostic_trace_event_gated_task3_seed44_fast_tokens8_test5_epoch30_best_20260706.csv
results/diagnostic_trace_age_gated_task3_seed44_fast_tokens8_val5_epoch24_best_20260706.csv
results/diagnostic_trace_age_gated_task3_seed44_fast_tokens8_test5_epoch24_best_20260706.csv
```

## Readout

Under the cheap 30-epoch, 5000-samples-per-epoch, 8-token diagnostic protocol, task 3 no longer
reproduces the older strong event-gated-over-age-gated result. Age-gated is slightly better on
validation and offline continuous MSE, and clearly better in held-out rollout.

```text
event-gated: continuous_mse 0.172217, held-out 1/9
age-gated:   continuous_mse 0.164281, held-out 4/9
```

This differs from the 2026-06-09 Kitchen4 from-scratch result, where the larger 20k-sample,
20-epoch seed-42 protocol gave event-gated 15/19 total and age-gated 5/19 total. The current
result says that positive event-gated task-3 behavior is not robust to this cheaper seed-44,
8-token diagnostic setting.
