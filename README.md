# DROID Trajectory Overlay Inference

Run the fine-tuned `pi05_droid_trajectory_overlay` checkpoint that follows trajectories drawn on the exterior camera image.

## How It Works

1. **Instruction decomposition** (GPT-4o-mini): Breaks task into manipulation steps
2. **Object detection** (Gemini Robotics-ER): Locates objects in camera image
3. **Trajectory generation** (GPT-4o-mini): Predicts manipulation trajectory waypoints
4. **Trajectory drawing**: Draws trajectory on `exterior_image_1_left` using the **exact same `TraceOverlayConfig` defaults** used during training (magenta `(255,0,255)`, thickness=1, no outline)
5. **Policy inference**: Sends annotated image to pi0.5 policy server
6. **Re-planning**: Every `plan_freq` steps, re-detects objects and updates trajectory

## Setup

### Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 1. Download the checkpoint

```bash
# Option A: Use huggingface-cli
huggingface-cli download brandonyang/pi05_droid_trajectory_overlay --local-dir checkpoints/pi05_droid_trajectory_overlay

# Option B: Use git
git clone https://huggingface.co/brandonyang/pi05_droid_trajectory_overlay checkpoints/pi05_droid_trajectory_overlay
```

### 2. Start the policy server (GPU machine)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_finetune \
    --policy.dir=checkpoints/pi05_droid_trajectory_overlay
```

### 3. Run inference (robot laptop)

```bash
uv run examples/droid_trajectory_overlay/main.py \
    --remote-host <policy_server_ip> \
    --remote-port 8000 \
    --external-camera left \
    --left-camera-id <id> \
    --right-camera-id <id> \
    --wrist-camera-id <id>
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--external-camera` | (required) | `"left"` or `"right"` — which external camera to use |
| `--remote-host` | `0.0.0.0` | Policy server IP |
| `--remote-port` | `8000` | Policy server port |
| `--open-loop-horizon` | `8` | Actions to execute per policy query |
| `--plan-freq` | `10` | Re-plan trajectory every N steps |
| `--max-plan-count` | `20` | Max replanning calls per rollout |
| `--gpt-model` | `gpt-4o-mini` | GPT model for trajectory generation |
| `--gemini-model` | `gemini-robotics-er-1.5-preview` | Gemini model for object detection |
| `--no-overlay` | `false` | Disable trajectory overlay (ablation) |
| `--save-dir` | `trajectory_overlay_runs` | Output directory for logs/videos |

## Output

Each run saves to `<save-dir>/<timestamp>/`:
- `instruction.txt` — the task instruction
- `traj_NNNN.jpg` — trajectory overlay visualization at each planning step
- `trajectories.json` — all predicted trajectories with metadata
- `rollout.mp4` — video of the rollout
- `result.txt` — success/failure and statistics

## TraceOverlayConfig (Training Defaults)

The trajectory is drawn with `TraceOverlayConfig()` defaults matching `build_dataset.py` from [branyang02/temp_drawing](https://github.com/branyang02/temp_drawing/tree/main/droid_trajectory):
- **Line**: red `(255,0,0)` → pink `(255,105,180)` gradient, thickness=3
- **Outline**: black `(0,0,0)`, thickness=5
- **Yellow dot**: radius=5, color `(255,255,0)`, black outline thickness=2. The dot position (`current_index`) advances each frame — during training this was set to the frame index `t`.
- **Spline interpolation**: none (`num_interpolated=0`) — raw trajectory points are used directly
- **No arrows, dashes, tick marks, or past trajectory**

Changing these will cause a train/test mismatch and degrade performance.
