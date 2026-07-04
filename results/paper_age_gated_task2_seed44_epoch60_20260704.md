# Task-2 Age-Gated ACT Seed 44 Epoch-60 Continuation

Date: 2026-07-04

The task-2 age-gated ACT seed-44 run was restored from the July 1 artifact backup and resumed
from epoch 58 to epoch 60.

```text
config: configs/paper_age_gated_act_task2_seed44_resume.yaml
checkpoint root: checkpoints/paper_age_gated_task2_seed44/age_gated_act_h20_task2_phase_memory_seed44
resume checkpoint: last.pt epoch 58
log: logs/paper_age_gated_task2_seed44_resume_59_60_20260704.log
```

Training completed cleanly:

```text
epoch 59: train_loss 0.025757, val_loss 0.048493, train_seconds 544.80, val_seconds 3.62
epoch 60: train_loss 0.024540, val_loss 0.061054, train_seconds 541.73, val_seconds 2.82
```

Checkpoint state after continuation:

```text
last.pt epoch: 60
last.pt val_mse: 0.0610540546476841
best.pt epoch: 56
best.pt/best_val: 0.044679350405931476
```

The epoch-60 validation loss is worse than the validation-selected epoch-56 `best.pt` and worse
than the epoch-58 continuation point.

Offline eval after the epoch-60 continuation:

```text
epoch-56 best.pt continuous_mse: 0.03577113375067711
epoch-56 best.pt continuous_mae: 0.12979439795017242
epoch-56 best.pt gripper_sign_accuracy: 0.9845625162124634

epoch-60 last.pt continuous_mse: 0.03525119759142399
epoch-60 last.pt continuous_mae: 0.12632304728031157
epoch-60 last.pt gripper_sign_accuracy: 0.9875625014305115
```

Epoch-56 `best.pt` rollout:

```text
train30 / val5 / test5 = 23/30, 3/5, 4/5 = 30/40
held-out val+test = 7/10
failure episode IDs:
  train: [6, 12, 24, 25, 32, 34, 38]
  val: [29, 40]
  test: [20]
```

Epoch-60 `last.pt` rollout:

```text
train30 / val5 / test5 = 23/30, 3/5, 4/5 = 30/40
held-out val+test = 7/10
failure episode IDs:
  train: [6, 8, 12, 14, 17, 26, 34]
  val: [29, 40]
  test: [7]
```

The earlier epoch-58 `last.pt` rollout was:

```text
train30 / val5 / test5 = 21/30, 3/5, 1/5 = 25/40
held-out val+test = 4/10
```

Both epoch 56 and epoch 60 improve substantially over the earlier epoch-58 rollout. Epoch 60 is
slightly stronger offline, while epoch 56 and epoch 60 are tied on aggregate rollout. Use either
epoch-56 `best.pt` for validation-selected reporting or epoch-60 `last.pt` for fixed-epoch reporting;
do not use the epoch-58 rollout as the final seed-44 age-gated result.
