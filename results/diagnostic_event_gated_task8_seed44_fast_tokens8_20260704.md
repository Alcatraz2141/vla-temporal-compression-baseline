# Task 8 Event-Gated Fast-Token Diagnostic, Seed 44

Date: 2026-07-04

Purpose: run a new-task event-gated diagnostic after the task-2 fast-token result showed age-gated strongly outperforming event-gated online.

Task:

```text
task id for rollout: 8
task filter: KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
```

Config:

```text
configs/diagnostic_event_gated_act_task8_seed44_fast_tokens8.yaml
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
```

Rationale:

`max_memory_tokens=8` may make the event-gated diagnostic harder because it reduces older-context samples from 64 to 32. However, the task-2 age-gated run used the same 8-token budget and reached 19/40 total with 6/10 held-out, while the task-2 event-gated run reached only 2/40 and 0/10 held-out. That matched result means compression alone does not explain the event-gated failure.

This task-8 run tests whether the current event gate fails broadly under the fast protocol or whether task 2 is unusually hostile to the event gate.

Status:

```text
not started at commit time
```
