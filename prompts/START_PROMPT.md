# Session Start Prompt

Use this verbatim when launching codex at the start of a new pod session.

---

You are a research agent for a VLA (Vision-Language-Action) temporal compression project.
Read AGENTS.md, experimentation.md, README.md, and libero_rollout_env/README.md fully before doing anything else.
These files contain the complete project state, all bugs fixed, all results so far, and what to do next.

Here is the current authoritative state so you know where to pick up:

**Completed:**
- All 6 training/rollout bugs fixed (temporal alignment, action history leakage, language mismatch, gripper, undertraining, stale chunks)
- Sliding window baseline: 50 epochs done. Best checkpoint epoch 18, val_mse 0.00847. Offline eval MSE 0.0593, MAE 0.294. Task-5 rollout: 0/1 success.
- Event-gated memory: 50 epochs done. Best checkpoint epoch 46, val_mse 0.00895. Offline eval and rollout NOT YET RUN.
- training/train.py now has a cosine LR scheduler with warmup — reads lr_schedule: cosine from config, no-op if absent.

**Immediately needed (do in this order):**
1. Run offline eval on event-gated 50ep best.pt:
   uv run python evaluation/eval.py --config configs/libero_long_event_gated_resume_last_to50.yaml --checkpoint checkpoints/libero_long/event_gated_memory/best.pt
2. Run task-5 online rollout on event-gated 50ep best.pt with video:
   bash libero_rollout_env/run_rollout.sh configs/libero_long_event_gated_resume_last_to50.yaml checkpoints/libero_long/event_gated_memory/best.pt --tasks 5 --episodes-per-task 1 --max-steps 300 --video-dir results/rollout_videos_event_gated_memory_50ep --video-every 1 --video-fps 20 --results-path results/libero_rollouts_event_gated_memory_50ep.csv
3. Train age-gated ablation (50 epochs, same recipe):
   uv run python train.py --config configs/ablation_gate_age.yaml
4. Train concat-query ablation (50 epochs):
   uv run python train.py --config configs/ablation_query_concat.yaml
5. Run offline eval and task-5 rollout for both ablations (same pattern as above)

**GPU and batch size:**
- Use batch_size: 64 on L40S (48GB) or A100
- num_workers: 8, prefetch_factor: 4
- lr: 2e-4 when using batch 64 (scaled from 1e-4 at batch 32)
- total_steps in config must match actual steps: (50000/64) * 50 = ~39000

**After each training run completes:**
- Save checkpoints, logs, results to HF backup immediately (do not wait until end)
- Backup command: see AGENTS.md restore section for the tar + hf upload pattern

**Do not:**
- Change any data loading code without checking episode_loader.py for side effects
- Retrain sliding window — it is done
- Change the loss function or model architecture for the ablations — only the gate_type and query_type differ

When done with all 4 steps, run the END prompt to document and save everything.
