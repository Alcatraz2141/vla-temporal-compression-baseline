# Task-2 Event Memory Gate Debug

Date: 2026-06-26

## Episode

```text
model: event_gated_act_h20_task2_phase_memory
checkpoint: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory/best.pt
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
split: train
episode index: 1
rollout result: success
steps: 248
```

## Artifacts

```text
video:
  results/memory_gate_debug/videos/event_gated_act_h20_task2_phase_memory/seed42_task02_episode1_KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.mp4

standard rollout trace:
  results/memory_gate_debug/trace_task2_demo1.csv

memory gate rows:
  results/memory_gate_debug/gates/event_gated_act_h20_task2_phase_memory/task02_episode1_memory_gates.csv

top-chunk summary:
  results/memory_gate_debug/task2_demo1_memory_gate_top_chunks.csv

contact sheet:
  results/memory_gate_debug/task2_demo1_memory_gate_contact_sheet.png
```

## Important Implementation Caveat

The current `event_gated_act` implementation does not hard-select top-k event chunks. For the
paper task-2 config:

```text
older context: 64 rollout steps
chunk_size: 4
max_memory_tokens: 16
```

This means the model keeps all 16 older-context chunk summaries and applies a learned scalar gate
to each summary. The visualization therefore shows the strongest soft-gated memory chunks and the
highest-attention frame inside each chunk, not exclusive retained/dropped frames.

## Observed Pattern

The strongest gate-weighted chunks concentrate around frames `85-92` for a long span of the
successful rollout:

```text
policy step 100: top chunks frames 89-92 and 85-88
policy step 120: top chunks frames 89-92 and 85-88
policy step 140: top chunks frames 89-92 and 85-88
policy step 160: top chunk frames 89-92
```

Later, after that older memory falls out of the 64-step window, the strongest chunks shift:

```text
policy step 200: top chunk frames 145-148
policy step 240: top chunks frames 229-232 and 225-228
```

The frame-85-to-92 focus is consistent with the model preserving a salient earlier interaction
state well after it is no longer in the recent 8-step context. This is useful qualitative evidence
that the memory path is using non-recent trajectory context.

## What This Does Not Prove Yet

This single successful rollout does not prove the event gate is better than the age-gated control.
To support that claim directly, run the same debug on:

```text
1. an age-gated checkpoint on the same episode,
2. an event-success / age-failure matched episode if available,
3. a phase-ACT failure / event-success episode from a positive comparison run.
```

The strongest paper evidence would be a side-by-side contact sheet showing event-gated memory
upweighting sub-goal or pre-occlusion frames while age-gated either tracks only recency or fails
to preserve that state.
