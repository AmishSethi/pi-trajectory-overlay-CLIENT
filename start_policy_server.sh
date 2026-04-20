#!/usr/bin/env bash
# Helper: start the pi0.5 trajectory-overlay policy server on the GPU machine.
#
# This script MUST be run from inside the openpi repo (https://github.com/Physical-Intelligence/openpi).
# Not from inside this (pi-trajectory-overlay) repo.
#
# Usage:
#   cd /path/to/openpi
#   bash /path/to/pi-trajectory-overlay/start_policy_server.sh [checkpoint_dir]

set -euo pipefail

CKPT_DIR="${1:-$HOME/checkpoints/pi05_droid_trajectory_overlay}"

if [[ ! -f "scripts/serve_policy.py" ]]; then
    echo "ERROR: scripts/serve_policy.py not found."
    echo "You must run this from inside the openpi repo, not pi-trajectory-overlay."
    echo "  git clone https://github.com/Physical-Intelligence/openpi.git"
    echo "  cd openpi"
    echo "  GIT_LFS_SKIP_SMUDGE=1 uv sync"
    echo "  bash /path/to/pi-trajectory-overlay/start_policy_server.sh"
    exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint directory not found: $CKPT_DIR"
    echo "Downloading from HuggingFace..."
    mkdir -p "$(dirname "$CKPT_DIR")"
    huggingface-cli download brandonyang/pi05_droid_trajectory_overlay \
        --local-dir "$CKPT_DIR"
fi

echo "Starting pi0.5 policy server on port 8000..."
echo "Checkpoint: $CKPT_DIR"
echo "Config: pi05_droid_finetune"
exec uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_finetune \
    --policy.dir="$CKPT_DIR"
