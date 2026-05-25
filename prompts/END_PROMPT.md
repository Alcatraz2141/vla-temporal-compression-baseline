# Session End Prompt

Use this verbatim before terminating a pod. Run it when all planned work for the session is done
or when you need to stop early and preserve state.

---

The session is ending. Do the following exactly in order before I terminate the pod. NO data will persist after i stop the Pod.

**Step 1 — Update AGENTS.md**
Add a new section at the bottom of AGENTS.md titled "## Current State as of <today's date>" containing:
- Which models have been trained and their best checkpoint info (epoch, val_mse, path)
- Which models have had offline eval run (MSE, MAE values)
- Which models have had online rollout run (success rate, video path)
- What is NOT done yet and what the next immediate commands are
- Any bugs or issues encountered this session and how they were resolved

**Step 2 — Update experimentation.md**
Add a new dated section (## YYYY-MM-DD <short description>) with:
- What was run this session (commands used)
- Key metrics observed (val_mse per epoch if notable, offline MSE/MAE, rollout results)
- Any config changes made and why
- GPU used, batch size, approximate time per epoch

**Step 3 — Update README.md**
Update the "Current Handoff" section to reflect the latest authoritative state:
- Latest completed checkpoint for each model
- What still needs to be run
- The restore sequence if anything changed (new HF dataset commit, etc.)

**Step 4 — Update libero_rollout_env/README.md**
If any rollout commands changed or new results came in, update the "current checkpoint" block at the top.

**Step 5 — Create checkpoint backup archive**
Run:
```bash
cd /root/vla-temporal-compression-baseline
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="vla_run_artifacts_${TIMESTAMP}.tar.gz"
tar -czf "/workspace/run_backups/${BACKUP_NAME}" \
    checkpoints/ \
    results/ \
    logs/ \
    splits/
echo "Created backup: ${BACKUP_NAME}"
```

**Step 6 — Push backup to Hugging Face**
Run:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run huggingface-cli upload Alcatraz1412/vla-run-backups \
    "/workspace/run_backups/${BACKUP_NAME}" \
    "${BACKUP_NAME}" \
    --repo-type dataset
echo "Uploaded to HF: ${BACKUP_NAME}"
```

**Step 7 — Git commit and push**
Stage and commit all changes to tracked files:
```bash
cd /root/vla-temporal-compression-baseline
git add AGENTS.md experimentation.md README.md libero_rollout_env/README.md
git add training/train.py  # if modified
git add configs/           # if any configs changed
git add prompts/           # always include
git status                 # show me what is staged before committing
git commit -m "session <date>: <one line summary of what was done>"
Do not Push i will do that myself
```

**Step 8 — Confirm**
Print a summary of what was saved:
- HF backup filename and URL
- Git commit hash
- List of models with their best val_mse
- What is still pending for the next session

Do not skip any step. If a step fails, note the error and continue with the rest. Update the handoffs properly. 
