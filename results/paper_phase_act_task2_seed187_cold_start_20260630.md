# Task-2 Phase ACT Cold-Start Seed 187

Date: 2026-06-30

```text
config: configs/paper_phase_act_task2_seed187_cold_start_resume.yaml
checkpoint: checkpoints/paper_phase_act_task2_seed187_cold_start/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
completed epoch: 60
best checkpoint by training validation: epoch 58
best val_mse: 0.02903650622618826
```

Offline eval on `best.pt`:

```text
continuous_mse: 0.03023523513815905
continuous_mae: 0.12312478846625279
gripper_sign_accuracy: 0.9893914555248461
pred_temporal_smoothness: 0.24471976725678696
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
train30: 25/30
val5: 4/5
test5: 4/5
total: 33/40
held-out val+test: 8/10
```

Failure episode IDs:

```text
train failures: [17, 21, 24, 32, 36]
val failures: [41]
test failures: [20]
```

Artifacts:

```text
offline eval log: logs/paper_phase_act_task2_seed187_cold_start_eval_20260630.log
rollout logs:
  logs/paper_phase_act_task2_seed187_cold_start_rollout_train30_20260630.log
  logs/paper_phase_act_task2_seed187_cold_start_rollout_val5_20260630.log
  logs/paper_phase_act_task2_seed187_cold_start_rollout_test5_20260630.log
rollout CSVs:
  results/paper_rollouts_phase_act_task2_seed187_cold_start_train30.csv
  results/paper_rollouts_phase_act_task2_seed187_cold_start_val5.csv
  results/paper_rollouts_phase_act_task2_seed187_cold_start_test5.csv
```

Interpretation:

```text
This cold-start phase ACT seed-187 run is much stronger online than the epoch-50 event-gated
seed-187 checkpoint: 33/40 total and 8/10 held-out for phase ACT versus 20/40 total and 3/10
held-out for event-gated. Offline action prediction also favors phase ACT: continuous_mse
0.030235 for phase ACT versus 0.042146 for event-gated epoch 50.
```
