#!/usr/bin/env bash
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eva_jiani_env
cd ~/pi-trajectory-overlay
set -a; source .env; set +a

# Plan-once mode: Gemini + GPT fire exactly once at t=0, then the same
# trajectory is reused for the entire rollout. No re-planning, no step
# completion checks. Avoids repeated API hangs on flaky networks.
exec python -u main.py \
    --remote-host 127.0.0.1 --remote-port 8000 \
    --external-camera left \
    --left-camera-id 26368109 \
    --right-camera-id 25455306 \
    --wrist-camera-id 15512737 \
    --max-plan-count 1 \
    --plan-freq 999999 \
    --no-show-display \
    --save-dir ~/pi-trajectory-overlay/runs
