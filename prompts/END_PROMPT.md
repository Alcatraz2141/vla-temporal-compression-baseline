# Session End Prompt

Use this verbatim before terminating a RunPod session. Run it when all planned work for the
session is done or when you need to stop early and preserve state.

---

The session is ending. Do the following exactly in order before I terminate the pod. Assume no data will persist after the pod stops unless it is backed up.

## Step 1 - Update AGENTS.md

Add a new section at the bottom of `AGENTS.md` titled:

```text
## Current State as of <today's date>
```

Include:

- corrected-H1 model(s) trained this session;
- commands used;
- checkpoint paths and best checkpoint info;
- best `val_loss`;
- offline eval metrics: `mse`, `mae`, `continuous_mse`, `continuous_mae`, `gripper_sign_accuracy`;
- offline diagnostics: `first_action_mse_per_element`, `position_mse`, `rotation_mse`, `gripper_sign_accuracy`;
- whether any rollout was run, including success rate, CSV path, and video path;
- what is not done yet;
- exact next immediate commands;
- any bugs, OOMs, dataloader stalls, driver/CUDA issues, or config changes.

## Step 2 - Update experimentation.md

Add a new dated section:

```text
## YYYY-MM-DD <short description>
```

Include:

- what was run this session;
- exact commands used;
- key train/validation metrics;
- offline eval and diagnostics metrics;
- rollout results if any;
- config changes and why;
- GPU type, VRAM, batch size, num_workers, prefetch_factor;
- approximate throughput or time per epoch/step if observed;
- decision against the RunPod gate:
  - did `val_loss` beat the local baseline around `0.744`?
  - did `gripper_sign_accuracy` beat the local baseline `0.555`?
  - should we continue, pause, or run event-gated next?

## Step 3 - Update README.md

Update the `Current LIBERO Handoff` section with the latest authoritative state:

- latest corrected-H1 checkpoint for each model run;
- whether the bounded sliding-window gate passed;
- whether event-gated corrected-H1 has been run;
- what still needs to be run;
- any changed restore/setup sequence.

## Step 4 - Update docs/libero_rollout_improvement_plan.md

Update the 2026-05-31 local-progress/RunPod-gate section if the decision changed.

Record:

- whether the bounded RunPod gate passed;
- latest threshold numbers;
- next decision rule;
- whether rollout should be attempted next.

## Step 5 - Update libero_rollout_env/README.md

If any rollout commands changed or new rollout results came in, update the current checkpoint/rollout block.

If no rollout was run, explicitly say that rollout remains pending until offline diagnostics pass.

## Step 6 - Create Checkpoint Backup Archive

Run:

```bash
cd /root/vla-temporal-compression-baseline
mkdir -p /workspace/run_backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="vla_run_artifacts_${TIMESTAMP}.tar.gz"
tar -czf "/workspace/run_backups/${BACKUP_NAME}" \
  checkpoints/ \
  results/ \
  logs/ \
  configs/ \
  splits/ \
  AGENTS.md \
  README.md \
  experimentation.md \
  docs/ \
  prompts/ \
  libero_rollout_env/ \
  utils/ \
  evaluation/ \
  scripts/
echo "Created backup: ${BACKUP_NAME}"
```

Do not include `data/libero_long` in the backup.

## Step 7 - Push Backup To Hugging Face

Run:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf upload Alcatraz1412/vla-run-backups \
  "/workspace/run_backups/${BACKUP_NAME}" \
  "${BACKUP_NAME}" \
  --repo-type dataset
echo "Uploaded to HF: ${BACKUP_NAME}"
```

If `hf upload` is unavailable, use the installed Hugging Face CLI equivalent and note the command used.

## Step 8 - Git Commit But Do Not Push

Stage and commit relevant code/config/doc changes:

```bash
cd /root/vla-temporal-compression-baseline
git add AGENTS.md README.md experimentation.md docs/ libero_rollout_env/README.md prompts/
git add configs/ training/ datasets/ models/ evaluation/ scripts/ utils/
git status
git commit -m "session <date>: <one line summary>"
```

Do not push. The user will push manually.

Do not stage:

```text
data/libero_long
large checkpoint binaries unless explicitly requested
large rollout videos unless intentionally tracked
```

## Step 9 - Confirm

Print a final summary containing:

- HF backup filename;
- Git commit hash;
- model/checkpoint best `val_loss`;
- eval/diagnostics metrics;
- whether the bounded gate passed;
- exact next command for the following session.

If any step fails, note the error and continue with the remaining steps. Do not skip handoff updates.
