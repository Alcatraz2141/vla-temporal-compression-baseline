#!/usr/bin/env bash
# =============================================================================
# setup_pod.sh — One-shot RunPod setup for vla-temporal-compression-baseline
# Usage: bash setup_pod.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}===> $1${NC}"; }
warn() { echo -e "${YELLOW}WARN: $1${NC}"; }
die()  { echo -e "${RED}ERROR: $1${NC}"; exit 1; }

# ── 0. Detect working dir ────────────────────────────────────────────────────
REPO_DIR="/root/vla-temporal-compression-baseline"
WORKSPACE="/workspace"
DATA_DIR="${WORKSPACE}/vla-temporal-compression-baseline-data"

step "Setting up workspace directories"
mkdir -p "${DATA_DIR}/libero_long"
mkdir -p "${WORKSPACE}/run_backups"

# ── 1. Clone repo if not already present ─────────────────────────────────────
step "Cloning repo"
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone https://github.com/Alcatraz2141/vla-temporal-compression-baseline.git "${REPO_DIR}"
else
    warn "Repo already exists at ${REPO_DIR}, pulling latest"
    git -C "${REPO_DIR}" pull
fi
cd "${REPO_DIR}"

# ── 2. Install uv ────────────────────────────────────────────────────────────
step "Installing uv"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# ── 3. Python env sync ───────────────────────────────────────────────────────
step "Syncing Python environment"
uv sync
uv add h5py hf-transfer

# ── 4. Node.js + codex ───────────────────────────────────────────────────────
step "Installing Node.js 20 and codex"
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
node -v && npm -v
npm install -g @openai/codex

# ── 5. LIBERO rollout environment ────────────────────────────────────────────
step "Bootstrapping LIBERO rollout env"
if [ -f "${REPO_DIR}/libero_rollout_env/bootstrap.sh" ]; then
    bash "${REPO_DIR}/libero_rollout_env/bootstrap.sh" || warn "LIBERO bootstrap had issues, check manually"
fi

# ── 6. Symlink data dir ──────────────────────────────────────────────────────
step "Symlinking data directory"
if [ ! -e "${REPO_DIR}/data/libero_long" ]; then
    mkdir -p "${REPO_DIR}/data"
    ln -s "${DATA_DIR}/libero_long" "${REPO_DIR}/data/libero_long"
    echo "  Symlinked data/libero_long -> ${DATA_DIR}/libero_long"
else
    warn "data/libero_long already exists, skipping symlink"
fi

# ── 7. Hugging Face auth (manual) ────────────────────────────────────────────
step "Hugging Face authentication (manual step required)"
echo ""
echo "  Run this to authenticate:"
echo "    uv run hf auth login"
echo ""
echo "  Press ENTER after you have authenticated to continue..."
read -r

# ── 8. Restore latest checkpoint backup from HF ──────────────────────────────
step "Downloading checkpoint backups from HF"
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run huggingface-cli download Alcatraz1412/vla-run-backups \
    --repo-type dataset \
    --local-dir "${WORKSPACE}/run_backups" || warn "Backup download failed, continuing"

LATEST_BACKUP="$(ls -t ${WORKSPACE}/run_backups/vla_run_artifacts_*.tar.gz 2>/dev/null | head -1 || true)"
if [ -n "${LATEST_BACKUP}" ]; then
    echo "  Restoring from: ${LATEST_BACKUP}"
    tar -xzf "${LATEST_BACKUP}" -C "${REPO_DIR}"
    echo "  Checkpoints restored"
else
    warn "No backup archives found in ${WORKSPACE}/run_backups"
fi

# ── 9. Download LIBERO data ──────────────────────────────────────────────────
step "Downloading LIBERO-Long dataset from HF"
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --local-dir "${DATA_DIR}/libero_long" \
    --include "libero_10/*.hdf5" \
    --max-workers 4 || warn "LIBERO download failed, check HF auth"

# ── 10. Validate setup ───────────────────────────────────────────────────────
step "Validating setup"
cd "${REPO_DIR}"
uv run python scripts/inspect_libero.py --data-root data/libero_long || warn "inspect_libero check failed"
uv run python scripts/smoke_test.py --sources libero_long || warn "smoke_test failed"

# ── 11. Codex auth (manual) ──────────────────────────────────────────────────
step "Codex authentication (manual step required)"
echo ""
echo "  Run this to authenticate codex:"
echo "    codex login --device-auth"
echo ""
echo "  After authenticating, launch your agent with:"
echo "    cd ${REPO_DIR} && codex \"<your starting prompt>\""
echo ""

step "Setup complete!"
echo "  Repo:        ${REPO_DIR}"
echo "  Data:        ${DATA_DIR}/libero_long"
echo "  Checkpoints: ${REPO_DIR}/checkpoints/"
echo "  Backups:     ${WORKSPACE}/run_backups/"
