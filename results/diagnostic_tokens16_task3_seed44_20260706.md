# Task-3 Tokens16 Diagnostic, Seed 44

Date: 2026-07-06

## Protocol

```text
task id: 3
task: KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
seed: 44
samples_per_epoch: 5000
epochs: 30
batch_size: 32
chunk_size: 4
max_memory_tokens: 16
older context frames: 64
rollout protocol: split-aware val5/test5, max_steps 300, temporal ensemble
```

This is the tokens16 rerun motivated by the older positive Kitchen4 result. It is not directly
matched to the old run because the old positive run used seed 42 and 20k samples per epoch.

## Event-Gated Result

```text
config: configs/diagnostic_event_gated_act_task3_seed44_tokens16.yaml
checkpoint: checkpoints/diagnostic_event_gated_task3_seed44_tokens16/diagnostic_event_gated_act_h20_task3_tokens16_seed44/best.pt
best epoch: 24
best val_loss: 0.1452547274529934

offline continuous_mse: 0.1961492970585823
offline continuous_mae: 0.29638053973515827
gripper_sign_accuracy: 0.9541666706403097

rollout val5: 0/4
rollout test5: 0/5
held-out val+test: 0/9
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task3_seed44_tokens16_val5_epoch24_best_20260706.csv
results/diagnostic_rollouts_event_gated_task3_seed44_tokens16_test5_epoch24_best_20260706.csv
results/diagnostic_trace_event_gated_task3_seed44_tokens16_val5_epoch24_best_20260706.csv
results/diagnostic_trace_event_gated_task3_seed44_tokens16_test5_epoch24_best_20260706.csv
```

## Age-Gated Status

```text
config: configs/diagnostic_age_gated_act_task3_seed44_tokens16.yaml
log: logs/diagnostic_age_gated_task3_seed44_tokens16_20260706.log
status when this summary was written: still training
latest completed epoch observed: 15
best epoch observed: 15
best val_loss observed: 0.19999533146619797
```

Age-gated tokens16 needs offline eval and val/test rollout after training completes.

## Interpretation

The event-gated tokens16 rerun did not recover the old positive Kitchen4 behavior under the
current cheap seed-44 protocol. It is worse than the task-3 tokens8 event-gated run both offline
and online:

```text
event-gated tokens8:  continuous_mse 0.17221714556217194, held-out 1/9
event-gated tokens16: continuous_mse 0.1961492970585823, held-out 0/9
```

The old positive Kitchen4 result remains most plausibly explained by the combined protocol
difference: seed 42, 20k samples per epoch, and tokens16/64 older frames. Tokens16 alone did not
reproduce it at seed 44 with 5k samples per epoch.
