# DROID Trajectory Overlay Inference

Standalone inference pipeline for the `brandonyang/pi05_droid_trajectory_overlay` checkpoint — a pi0.5 model fine-tuned to follow trajectories drawn on the exterior camera image.

## Architecture

This repo runs the **robot laptop side** of the inference loop. The pi0.5 model itself runs on a separate **GPU server** using the [openpi](https://github.com/Physical-Intelligence/openpi) repository.

```
┌─────────────────────┐      websocket       ┌──────────────────────┐
│   GPU server        │ ◄─────────────────► │  Robot laptop        │
│   (openpi repo)     │                       │  (THIS repo)         │
│                     │                       │                      │
│  serve_policy.py    │                       │  main.py             │
│  pi0.5 checkpoint   │  ◄── annotated img ── │  + trajectory overlay│
│                     │  ──── actions ────►   │  + DROID robot I/O   │
└─────────────────────┘                       └──────────────────────┘
```

## Pipeline

1. **Instruction decomposition** (GPT-4o-mini): Breaks task into manipulation steps
2. **Object detection** (Gemini Robotics-ER): Locates objects in camera image
3. **Trajectory generation** (GPT-4o-mini): Predicts 10-15 manipulation waypoints
4. **Trajectory drawing** (local): Draws trajectory on `exterior_image_1_left` using training-matched `TraceOverlayConfig` (red→pink gradient, yellow dot, black outline)
5. **Policy inference**: Sends annotated 224×224 image to pi0.5 policy server via websocket
6. **Action execution**: Runs returned actions on the Franka via DROID's `RobotEnv`
7. **Re-planning**: Every `plan_freq` steps, re-detects objects and updates trajectory

## ─── Robot Laptop Setup (this repo) ───

### 1. Clone and install

```bash
git clone https://github.com/AmishSethi/pi-trajectory-overlay.git
cd pi-trajectory-overlay
pip install -r requirements.txt
```

You also need [DROID's robot infrastructure](https://github.com/droid-dataset/droid) installed on the laptop — that provides `droid.robot_env.RobotEnv` which talks to the Franka.

### 2. Set API keys

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file (git-ignored) with those two lines.

### 3. Run inference

```bash
python main.py \
    --remote-host <GPU_SERVER_IP> \
    --remote-port 8000 \
    --external-camera left \
    --left-camera-id <your_zed_left_serial> \
    --right-camera-id <your_zed_right_serial> \
    --wrist-camera-id <your_zed_wrist_serial>
```

You will be prompted for a task instruction (e.g. "Pick up the red block and put it in the bowl"). Press Ctrl+C or `q` in the display window to stop early.

## ─── GPU Server Setup (openpi repo) ───

The policy server is part of the [openpi](https://github.com/Physical-Intelligence/openpi) repository, not this one. On the GPU machine:

### 1. Clone and install openpi

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. Download the trajectory-overlay checkpoint

```bash
# From anywhere on the GPU machine
huggingface-cli download brandonyang/pi05_droid_trajectory_overlay \
    --local-dir ~/checkpoints/pi05_droid_trajectory_overlay
```

### 3. Start the policy server

```bash
# From inside the openpi repo
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_finetune \
    --policy.dir=~/checkpoints/pi05_droid_trajectory_overlay
```

The server listens on port 8000 by default. Make sure the GPU machine is reachable from the robot laptop.

## Key Arguments (`main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--external-camera` | (required) | `"left"` or `"right"` — which external camera to use |
| `--remote-host` | `0.0.0.0` | Policy server IP |
| `--remote-port` | `8000` | Policy server port |
| `--open-loop-horizon` | `8` | Actions executed per policy query |
| `--plan-freq` | `10` | Re-plan trajectory every N steps |
| `--max-plan-count` | `20` | Max replanning calls per rollout |
| `--trajectory-source` | `gpt` | `gpt`, `retrieval`, or `fallback` |
| `--gpt-model` | `gpt-4o-mini` | OpenAI model for trajectory generation |
| `--gemini-model` | `gemini-robotics-er-1.5-preview` | Gemini model for object detection |
| `--no-overlay` | `false` | Disable trajectory overlay (ablation) |
| `--save-dir` | `trajectory_overlay_runs` | Output directory |
| `--show-display` | `true` | Show live OpenCV debug window |

## Trajectory Sources

This repo supports two trajectory generation methods:

- **`gpt`** (default): Gemini detects objects, GPT plans 10-15 waypoints. Fast (~5s/plan), no preprocessed data needed.
- **`retrieval`**: Retrieves the most similar DROID episode, warps its SAM2 trajectory using Gemini correspondence. Slower (~60s/plan), requires preprocessed DROID 5K data. See `trajectory_source.py` for paths.
- **`fallback`**: Try `retrieval` first, fall back to `gpt` on failure.

## Output

Each run saves to `<save-dir>/<timestamp>/`:
- `instruction.txt` — the task instruction
- `traj_NNNN.jpg` — trajectory overlay visualization at each planning step
- `trajectories.json` — all predicted trajectories with metadata
- `rollout.mp4` — video of the rollout
- `result.txt` — success/failure and statistics
- `viz/debug_frames/` — per-frame debug panels (raw + annotated + wrist + info)

## TraceOverlayConfig (Training Defaults)

The trajectory is drawn with `TraceOverlayConfig()` defaults matching `build_dataset.py` from [branyang02/temp_drawing](https://github.com/branyang02/temp_drawing/tree/main/droid_trajectory):
- **Line**: red `(255,0,0)` → pink `(255,105,180)` gradient, thickness=3
- **Outline**: black `(0,0,0)`, thickness=5
- **Yellow dot**: radius=5, color `(255,255,0)`, black outline thickness=2. Position (`current_index`) advances each frame.
- **Spline interpolation**: `num_interpolated=100` at inference (sparse GPT waypoints → dense visual curve)
- **No arrows, dashes, tick marks, or past trajectory**

Changing these will cause train/test mismatch and degrade policy performance.

## Tests

```bash
python -m pytest tests/ -v
```

97 unit tests covering drawing, coordinate transforms, API mocking, and pipeline simulation.

## Batch Evaluation

To evaluate trajectory quality across many DROID episodes without a real robot:

```bash
python batch_evaluate.py --start 0 --end 100 --output-dir batch_results/run1
```

Produces per-episode quality classification (good/partial/fail) and a summary JSON.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Standalone inference entry point — runs on robot laptop |
| `traj_vis_utils.py` | Trajectory drawing with training-matched `TraceOverlayConfig` |
| `trajectory_predictor.py` | Gemini object detection + GPT waypoint generation |
| `trajectory_source.py` | Unified interface for `gpt` / `retrieval` / `fallback` sources |
| `inference_visualizer.py` | Real-time OpenCV debug display |
| `generate_examples.py` | Gallery generator for visual examples |
| `batch_evaluate.py` | Batch evaluation harness |
| `tests/` | 97 unit tests |
