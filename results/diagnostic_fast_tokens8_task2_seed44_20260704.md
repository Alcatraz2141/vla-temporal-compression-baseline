# Task 2 Fast Memory-Token Diagnostic, Seed 44

Date: 2026-07-04

Purpose: test a faster matched event-gated vs age-gated ACT diagnostic by reducing older memory from 16 chunks to 8 chunks and reducing training samples per epoch.

Shared settings:

```text
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
seed: 44
samples_per_epoch: 5000
epochs: 10
batch_size: 32
chunk_size: 4
max_memory_tokens: 8
older context frames: 32
cache_max_episodes: 128
num_workers: 8
prefetch_factor: 2
```

Configs:

```text
configs/diagnostic_event_gated_act_task2_seed44_fast_tokens8.yaml
configs/diagnostic_age_gated_act_task2_seed44_fast_tokens8.yaml
```

Training logs:

```text
logs/diagnostic_event_gated_task2_seed44_fast_tokens8_20260704.log
logs/diagnostic_age_gated_task2_seed44_fast_tokens8_20260704.log
```

## Event-Gated

```text
checkpoint: checkpoints/diagnostic_event_gated_task2_seed44_fast_tokens8/diagnostic_event_gated_act_h20_task2_fast_tokens8_seed44/best.pt
best epoch: 10
epoch-10 train_loss: 0.230463
epoch-10 val_loss: 0.289246
avg train seconds/epoch: 80.726
avg val seconds/epoch: 1.822
offline continuous_mse: 0.28163617551326753
offline continuous_mae: 0.37690114974975586
gripper_sign_accuracy: 0.8521250128746033
```

## Age-Gated

```text
checkpoint: checkpoints/diagnostic_age_gated_task2_seed44_fast_tokens8/diagnostic_age_gated_act_h20_task2_fast_tokens8_seed44/best.pt
best epoch: 10
epoch-10 train_loss: 0.250300
epoch-10 val_loss: 0.335321
avg train seconds/epoch: 79.320
avg val seconds/epoch: 1.700
offline continuous_mse: 0.32570928037166597
offline continuous_mae: 0.40283342003822326
gripper_sign_accuracy: 0.800125002861023
```

## Readout

Reducing `max_memory_tokens` from 16 to 8 reduced the older image volume from 64 frames to 32 frames per sample and produced roughly 80-second training epochs. This is about 6x faster than the recent full age-gated continuation epochs around 8-9 minutes/epoch, so this is a useful screening protocol.

In this short diagnostic, event-gated is better than age-gated on validation and offline action prediction:

```text
val_loss:       event 0.289246 vs age 0.335321
continuous_mse: event 0.281636 vs age 0.325709
gripper_acc:    event 0.852125 vs age 0.800125
```

This is not a rollout result and should not be used as a paper number. It is evidence that the reduced-token diagnostic is fast enough for screening, and that the event gate does not look worse than age gate under this shorter-context, short-training setup.

## Rollout

Rollout protocol:

```text
task: 2
max_steps: 300
temporal_ensemble: true
seed: 42
splits: train30 / val5 / test5
```

Rollout results:

```text
event-gated best.pt:
  train30: 1/30
  val5:    0/5
  test5:   0/5
  total:   1/40
  held-out val+test: 0/10
  train successes: [38]

age-gated best.pt:
  train30: 4/30
  val5:    0/5
  test5:   0/5
  total:   4/40
  held-out val+test: 0/10
  train successes: [5, 14, 25, 36]
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task2_seed44_fast_tokens8_train30_20260704.csv
results/diagnostic_rollouts_event_gated_task2_seed44_fast_tokens8_val5_20260704.csv
results/diagnostic_rollouts_event_gated_task2_seed44_fast_tokens8_test5_20260704.csv
results/diagnostic_rollouts_age_gated_task2_seed44_fast_tokens8_train30_20260704.csv
results/diagnostic_rollouts_age_gated_task2_seed44_fast_tokens8_val5_20260704.csv
results/diagnostic_rollouts_age_gated_task2_seed44_fast_tokens8_test5_20260704.csv
```

Rollout readout:

The 10-epoch fast checkpoints are too undertrained for closed-loop evaluation. Event-gated looks better offline, but age-gated has the stronger seen-train closed-loop signal. Both fail all held-out val/test starts, so this diagnostic should be used for speed and early offline screening only.

Next action: continue the same fast-token configs to a longer diagnostic budget, for example 30 epochs, before spending time on more rollout. If held-out is still zero at 30 epochs, the `max_memory_tokens=8` setting is too compressed or the short `samples_per_epoch=5000` budget is insufficient for rollout competence.

## Epoch-30 Continuation

Both fast-token diagnostics were resumed from epoch 10 to epoch 30.

Event-gated:

```text
resume config: configs/diagnostic_event_gated_act_task2_seed44_fast_tokens8_resume30.yaml
log: logs/diagnostic_event_gated_task2_seed44_fast_tokens8_resume11_30_20260704.log
completed epoch: 30
best epoch: 26
best val_loss: 0.1532006323337555
epoch-30 val_loss: 0.22111310362815856
best checkpoint: checkpoints/diagnostic_event_gated_task2_seed44_fast_tokens8/diagnostic_event_gated_act_h20_task2_fast_tokens8_seed44/best.pt
offline continuous_mse: 0.16484375
offline continuous_mae: 0.2903249144554138
gripper_sign_accuracy: 0.9565625190734863
```

Age-gated:

```text
resume config: configs/diagnostic_age_gated_act_task2_seed44_fast_tokens8_resume30.yaml
log: logs/diagnostic_age_gated_task2_seed44_fast_tokens8_resume11_30_20260704.log
completed epoch: 30
best epoch: 28
best val_loss: 0.14813608974218367
epoch-30 val_loss: 0.1536689528822899
best checkpoint: checkpoints/diagnostic_age_gated_task2_seed44_fast_tokens8/diagnostic_age_gated_act_h20_task2_fast_tokens8_seed44/best.pt
offline continuous_mse: 0.1457701027393341
offline continuous_mae: 0.2710952818393707
gripper_sign_accuracy: 0.9613124966621399
```

Epoch-30 best-checkpoint rollouts:

```text
event-gated epoch-26 best:
  train30: 2/30
  val5:    0/5
  test5:   0/5
  total:   2/40
  held-out val+test: 0/10
  train successes: [13, 36]

age-gated epoch-28 best:
  train30: 13/30
  val5:    4/5
  test5:   2/5
  total:   19/40
  held-out val+test: 6/10
  train successes: [1, 4, 12, 13, 17, 24, 25, 30, 31, 32, 33, 35, 36]
  val successes: [9, 29, 40, 41]
  test successes: [11, 20]
```

Epoch-30 rollout readout:

The longer fast-token diagnostic strongly favors age-gated memory. Age-gated recovered to useful rollout behavior by epoch 28 despite using only 8 memory tokens and 5000 samples per epoch. Event-gated improved offline relative to epoch 10 but remained essentially nonfunctional online, with 0/10 held-out and only 2/30 train successes.

This is another case where offline action prediction is not enough: event and age had similar validation loss, but rollout diverged sharply.
