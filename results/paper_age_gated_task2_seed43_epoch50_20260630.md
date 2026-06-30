# Task-2 Age-Gated ACT Seed 43 Epoch-50 Audit

Date: 2026-06-30

```text
config: configs/paper_age_gated_act_task2_seed43_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed43/age_gated_act_h20_task2_phase_memory_seed43/last.pt
stopped after completed epoch: 50
epoch-50 val_mse: 0.04710079990327358
best_val: 0.04710079990327358
```

Offline eval:

```text
continuous_mse: 0.038894045352935794
continuous_mae: 0.13535064458847046
gripper_sign_accuracy: 0.9896250009536743
pred_temporal_smoothness: 0.2244861751794815
```

Rollout protocol:

```text
suite: LIBERO-10
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
max_steps: 300
temporal_ensemble: true
rollout_seed: 42
split protocol: train30 / val5 / test5
```

Rollout result:

```text
train30: 20/30
val5: 4/5
test5: 3/5
total: 27/40
held-out val+test: 7/10
```

Failure episode IDs:

```text
train failures: [0, 4, 8, 12, 14, 19, 27, 28, 35, 37]
val failures: [40]
test failures: [7, 20]
```

Artifacts:

```text
training logs:
  logs/paper_age_gated_task2_seed43_2epoch_20260630.log
  logs/paper_age_gated_task2_seed43_resume_20260630.log
offline eval log:
  logs/paper_age_gated_task2_seed43_eval_epoch50_20260630.log
rollout logs:
  logs/paper_age_gated_task2_seed43_rollout_train30_epoch50_20260630.log
  logs/paper_age_gated_task2_seed43_rollout_val5_epoch50_20260630.log
  logs/paper_age_gated_task2_seed43_rollout_test5_epoch50_20260630.log
rollout CSVs:
  results/paper_rollouts_age_gated_task2_seed43_train30_epoch50.csv
  results/paper_rollouts_age_gated_task2_seed43_val5_epoch50.csv
  results/paper_rollouts_age_gated_task2_seed43_test5_epoch50.csv
```

Comparison:

```text
age-gated ACT seed 43 epoch 50:
  offline continuous_mse: 0.038894045352935794
  rollout train30 / val5 / test5: 20/30, 4/5, 3/5 = 27/40
  held-out val+test: 7/10

event-gated ACT seed 43 from scratch:
  offline continuous_mse: 0.01848969299942255
  rollout train30 / val5 / test5: 18/30, 3/5, 3/5 = 24/40
  held-out val+test: 6/10

cold-start phase ACT seed 43:
  offline continuous_mse: 0.030674645383107036
  rollout train30 / val5 / test5: 18/30, 4/5, 1/5 = 23/40
  held-out val+test: 5/10
```

Interpretation:

```text
Age-gated ACT seed 43 is stronger online than both event-gated ACT seed 43 and cold-start phase
ACT seed 43, despite worse offline action-prediction MSE. This is evidence that the current
event-gating score is not helping on task 2 and may be selecting or weighting older memory poorly.
It also reinforces that offline continuous_mse is not a reliable checkpoint or method selector
for closed-loop LIBERO success.
```
