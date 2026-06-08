# Task 2 Age-Gated ACT Control

Date: 2026-06-08

Purpose:

```text
Control whether task-2 memory gains come from generic older-context memory/recency,
or from event-aware memory selection.
```

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Config:

```text
configs/libero_long_age_gated_act_h20_task2_phase_memory20.yaml
```

Checkpoint:

```text
checkpoints/libero_long_corrected_task2/age_gated_act_h20_task2_phase_memory20/last.pt
```

Training:

```text
resume source: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
resume source epoch: 29
continued epochs: 30..49
checkpoint used: last.pt from epoch 49
epoch 49 train_loss: 0.031259
epoch 49 lightweight val_loss: 0.154080
```

Note:

```text
The age-gated training config used val_split=val and eval_windows_per_episode=1 to avoid
the train-split validation bottleneck. The standard eval.py row in results/baselines_corrected_task2.csv
therefore reflects the checkpoint-embedded lightweight validation config and should not be compared
directly against the earlier train-split offline eval rows. The rollout results below are the main
comparison.
```

Rollout result:

| Split | Phase ACT original | Phase ACT continued20 | Age-gated ACT | Event-gated ACT |
|---|---:|---:|---:|---:|
| train10 | 9/10 | 6/10 | 3/10 | 10/10 |
| val5 | 2/5 | 3/5 | 2/5 | 5/5 |
| test5 | 4/5 | 3/5 | 0/5 | 4/5 |
| total | 15/20 | 12/20 | 5/20 | 19/20 |
| held-out val+test | 6/10 | 6/10 | 2/10 | 9/10 |

Failures:

```text
age-gated train failures:
  0, 4, 5, 6, 10, 12, 14

age-gated val failures:
  3, 40, 41

age-gated test failures:
  2, 7, 11, 20, 22
```

Interpretation:

```text
This is a strong negative age-gated result.

Age-based memory does not explain the task-2 event-gated improvement. Under the same task,
same phase-ACT warm start, and same 20-epoch continuation budget, age-gated ACT collapses
to 5/20 total and 0/5 on test. Event-gated ACT remains 19/20 total and 4/5 on test.

The result supports the claim that event-aware memory selection matters more than simply
adding older-context memory with a recency/age prior.
```

Artifacts:

```text
logs/age_gated_act_h20_task2_phase_memory20_20260608_valsplit.log
results/libero_rollouts_age_gated_act_task2_train10.csv
results/libero_rollouts_age_gated_act_task2_val5.csv
results/libero_rollouts_age_gated_act_task2_test5.csv
results/rollout_trace_age_gated_act_task2_train10.csv
results/rollout_trace_age_gated_act_task2_val5.csv
results/rollout_trace_age_gated_act_task2_test5.csv
```
