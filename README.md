# pi-trajectory-overlay

Standalone DROID inference client for pi0.5 checkpoints — with optional
Gemini/GPT trajectory-overlay guidance.

## Architecture

```
┌──────────────────┐   websocket   ┌───────────────────┐
│  GPU host        │ ◄──────────► │  Robot laptop     │
│  openpi server   │               │  main[_pi05].py   │
│  :8000           │               │  + Franka + ZEDs  │
└──────────────────┘               └───────────────────┘
```

Two client scripts share camera handling (ZED BGRA→RGB, 180° rotate externals,
wrist unrotated) and the same websocket protocol:

- **`main.py`** — GPT decomposition → Gemini ER object localization →
  GPT waypoints → red→pink overlay with yellow dot drawn on the exterior frame
  → sent to policy. Pair with an overlay-trained checkpoint (e.g.
  `brandonyang/pi05_droid_trajectory_overlay`, `ASethi04/pi05-chris-traces-v1`).
- **`main_pi05.py`** — raw frame + instruction as `prompt`, no LLM in the loop.
  Pair with vanilla pi0.5-droid (`gs://openpi-assets/checkpoints/pi05_droid`).

## Setup

```bash
pip install -r requirements.txt   # jax, openpi_client, cv2, google-genai, openai, tyro, ...
cat > .env <<'EOF'
GEMINI_API_KEY=<your_key>
OPENAI_API_KEY=<your_key>
EOF
chmod 600 .env
```
`.env` is git-ignored. Only `main.py` needs the keys.

DROID's `RobotEnv` and `pyzed` must be installed separately (usually via a
conda env; this rig uses `eva_jiani_env`).

## Run

Launch a policy server on a GPU box (openpi's `scripts/serve_policy.py`),
then on the laptop:

```bash
# LLM + trajectory-overlay client
python -u main.py \
    --remote-host <SERVER_IP> --remote-port 8000 \
    --external-camera left \
    --left-camera-id  <id> --right-camera-id <id> --wrist-camera-id <id> \
    --save-frames --save-dir runs

# No-LLM client (for vanilla pi0.5-droid)
python -u main_pi05.py \
    --remote-host <SERVER_IP> --remote-port 8000 \
    --external-camera left \
    --left-camera-id  <id> --right-camera-id <id> --wrist-camera-id <id> \
    --save-frames --save-dir runs_pi05
```

See `run_laptop.sh` for the launcher wrapper used on this rig.

## Key flags (`main.py`)

| Flag | Default | Effect |
|---|---|---|
| `--plan-freq` | 150 | re-plan every N env steps (Gemini + GPT, ~5–10 s each) |
| `--max-plan-count` | 20 | cap on re-plans per rollout |
| `--open-loop-horizon` | 8 | actions executed per policy query (chunk is 16) |
| `--save-frames` | false | dump 5 JPEGs/step to `<run>/frames/` — survives `pkill -9` |
| `--no-show-display` | — | disable OpenCV live window (needed for ssh/headless) |
| `--no-overlay` | false | ablation — don't draw trajectory |

`main_pi05.py` has the camera/remote flags plus `--save-frames`, no planning.

## Per-rollout outputs (`<save-dir>/<timestamp>/`)

`instruction.txt`, `actions.log` (per-step), `rollout.mp4` (graceful exit),
`result.txt` (`y/n` prompt). With `--save-frames`:
`frames/NNNN_{ext_raw,ext_annotated,wrist,ext_224,wrist_224}.jpg`. `main.py`
additionally writes `traj_NNNN.jpg`, `planning_report_NNNN.txt`,
`trajectories.json`, `viz/{debug_frames,raw_frames,annotated_frames}/`.

## Kill

```bash
tmux kill-session -t rollout 2>/dev/null; pkill -9 -f pi-trajectory-overlay/main
```
Same command for both clients — session name is always `rollout`, the `main`
prefix matches `main.py` and `main_pi05.py`.

## Tests

```bash
python -m pytest tests/
```
**154 tests**: unit (drawing, image pipeline, schema parsing, camera flip,
plan-once merge, IPv4 patch), live ZED hardware, real Gemini + real OpenAI.
Hardware/API tests skip automatically if unavailable.

## Notes

- **Trajectory overlay is stable within a plan window.** Waypoints and the
  yellow-dot `current_index` do not move until the next `--plan-freq` fires.
  Advancing by time would collapse the drawn line regardless of real progress.
- **IPv4-only DNS patch.** `trajectory_predictor.py` monkey-patches
  `socket.getaddrinfo` at import to strip AF_INET6 — UPenn SEAS silently drops
  IPv6 replies, making Gemini calls hang for 60–90 s otherwise. Set
  `ALLOW_IPV6=1` to disable.
- **Re-planning reuses the cached end point** (`PlanningState.current_end_point`)
  so the target stays stable while the manipulating object's position updates
  each plan.
- **Instruction-decomposition cache** lives at `<save-dir>/instruction_cache.json`
  — repeat tasks skip the GPT decomposition call.
