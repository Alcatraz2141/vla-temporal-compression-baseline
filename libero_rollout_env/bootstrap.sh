#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# bootstrap.sh — One-shot setup for LIBERO online rollout on RunPod
#
# Hardware assumptions:
#   GPU:    RTX 4090 (24 GB VRAM)
#   Volume: /workspace  (120 GB, persistent)
#   Disk:   /root       (20 GB, ephemeral)
#
# Usage:
#   cd /root/vla-temporal-compression-baseline
#   bash libero_rollout_env/bootstrap.sh
#
# After this finishes you can run rollouts with:
#   bash libero_rollout_env/run_rollout.sh <config> <checkpoint> [extra args]
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="/workspace"
ENVS_DIR="${WORKSPACE}/libero_rollout_envs"
VENV_DIR="${ENVS_DIR}/.venv"
LIBERO_SRC="${ENVS_DIR}/LIBERO"
DATA_DIR="${WORKSPACE}/vla-temporal-compression-baseline-data"
export UV_CACHE_DIR="${WORKSPACE}/uv-cache"
mkdir -p "${UV_CACHE_DIR}"

echo "=== [1/7] System packages for headless rendering ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    libegl1 libgl1 libosmesa6 libosmesa6-dev \
    libglew-dev patchelf cmake > /dev/null 2>&1
echo "    done."

echo "=== [2/7] Install uv if missing ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "    uv $(uv --version)"

echo "=== [3/7] Clone LIBERO source ==="
mkdir -p "${ENVS_DIR}"
if [ ! -d "${LIBERO_SRC}/.git" ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_SRC}"
else
    echo "    LIBERO already cloned."
fi

echo "=== [4/7] Create isolated uv venv ==="
if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}" --python 3.11
fi
echo "    venv at ${VENV_DIR}"

echo "=== [5/7] Install dependencies into isolated venv ==="
pip_install() {
    VIRTUAL_ENV="${VENV_DIR}" uv pip install --python "${VENV_DIR}/bin/python" "$@"
}

# Step 1: torch + torchvision from CUDA 12.4 index
pip_install \
    "torch>=2.6.0,<2.7" \
    "torchvision>=0.21.0,<0.22" \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: exact pins matching the known-working setup from experimentation.md
# plus LIBERO's transitive deps that its setup.py doesn't declare
pip_install \
    "numpy==1.26.4" \
    "mujoco==3.8.1" \
    "robosuite==1.4.0" \
    "bddl==1.0.1" \
    "braceexpand>=0.1.7" \
    "gym==0.25.2" \
    "robomimic==0.2.0" \
    "h5py>=3.14.0" \
    "pyyaml>=6.0.3" \
    "tqdm>=4.67.3" \
    "opencv-python>=4.10.0" \
    "pyopengl>=3.1.7" \
    "future>=1.0.0" \
    "hydra-core>=1.2.0" \
    "einops>=0.4.1" \
    "cloudpickle>=2.1.0" \
    "easydict>=1.9" \
    "matplotlib>=3.5.3"

# Step 3: install LIBERO editable (--no-deps since setup.py has no install_requires
# and we already installed everything above; avoids accidental version overrides)
pip_install --no-deps -e "${LIBERO_SRC}"

echo "    deps installed."

echo "=== [6/7] Write LIBERO config ==="
LIBERO_CONFIG_DIR="${REPO_ROOT}/libero_config"
mkdir -p "${LIBERO_CONFIG_DIR}"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
assets: ${LIBERO_SRC}/libero/libero/assets
bddl_files: ${LIBERO_SRC}/libero/libero/bddl_files
benchmark_root: ${LIBERO_SRC}/libero/libero
datasets: ${DATA_DIR}/libero_sim_datasets
init_states: ${LIBERO_SRC}/libero/libero/init_files
YAML
echo "    libero_config/config.yaml written."

echo "=== [7/7] Download LIBERO-Long data ==="
mkdir -p "${DATA_DIR}/libero_long"
# Symlink so the repo sees it at data/libero_long
ln -sfn "${DATA_DIR}/libero_long" "${REPO_ROOT}/data/libero_long"

# Use the main project venv for hf download (it has hf-transfer)
if [ ! -f "${DATA_DIR}/libero_long/libero_10/.download_done" ]; then
    echo "    Downloading LIBERO-Long HDF5 files..."
    cd "${REPO_ROOT}"
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
        --repo-type dataset \
        --local-dir "${DATA_DIR}/libero_long" \
        --include "libero_10/*.hdf5" \
        --max-workers 2
    touch "${DATA_DIR}/libero_long/libero_10/.download_done"
    echo "    download complete."
else
    echo "    LIBERO data already present, skipping download."
fi

echo ""
echo "=============================================="
echo "  LIBERO rollout environment ready!"
echo ""
echo "  Venv:   ${VENV_DIR}"
echo "  LIBERO: ${LIBERO_SRC}"
echo "  Data:   ${DATA_DIR}/libero_long"
echo ""
echo "  Quick test:"
echo "    bash libero_rollout_env/run_rollout.sh \\"
echo "      configs/libero_long_sliding_window.yaml \\"
echo "      checkpoints/libero_long/sliding_window/best.pt \\"
echo "      --tasks 0 --episodes-per-task 1 --max-steps 50"
echo "=============================================="
