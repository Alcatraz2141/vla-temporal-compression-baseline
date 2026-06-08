# Task-5 Event-Memory Confirmation

Date: 2026-06-08

Protocol:

```text
task: 5
suite: libero_10
max_steps: 300
temporal_ensemble: enabled
video: disabled
trace_csv: enabled
train split requested: 20 demos
val split requested: 10 demos, available: 5 demos
test split requested: 10 demos, available: 5 demos
```

The existing task-5 split contains 40 train, 5 val, and 5 test demos. The confirmation therefore used the larger feasible split-aware protocol:

```text
train20 / val5 / test5
```

## Results

| model | train | val | test | total | held-out |
|---|---:|---:|---:|---:|---:|
| phase ACT | 15/20 | 4/5 | 4/5 | 23/30 | 8/10 |
| phase + event-gated memory ACT | 20/20 | 4/5 | 5/5 | 29/30 | 9/10 |

Interpretation:

```text
The positive task-5 memory result strengthened under the larger feasible confirmation.
Event-gated ACT improved total rollout success from 23/30 to 29/30.
The largest gain came from train-init robustness: 15/20 -> 20/20.
Held-out success also improved slightly: 8/10 -> 9/10.
```

Per-demo failures:

```text
phase ACT train failures: 8, 10, 11, 18, 20
phase ACT val failure:    34
phase ACT test failure:   6

event-gated ACT train failures: none
event-gated ACT val failure:    36
event-gated ACT test failures:  none
```

Result files:

```text
results/libero_rollouts_phase_act_task5_train20_confirm.csv
results/libero_rollouts_phase_act_task5_val5_confirm.csv
results/libero_rollouts_phase_act_task5_test5_confirm.csv
results/libero_rollouts_event_gated_act_task5_train20_confirm.csv
results/libero_rollouts_event_gated_act_task5_val5_confirm.csv
results/libero_rollouts_event_gated_act_task5_test5_confirm.csv
```

Trace files:

```text
results/rollout_trace_phase_act_task5_train20_confirm.csv
results/rollout_trace_phase_act_task5_val5_confirm.csv
results/rollout_trace_phase_act_task5_test5_confirm.csv
results/rollout_trace_event_gated_act_task5_train20_confirm.csv
results/rollout_trace_event_gated_act_task5_val5_confirm.csv
results/rollout_trace_event_gated_act_task5_test5_confirm.csv
```
