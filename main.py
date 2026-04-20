"""
Standalone DROID inference with trajectory overlay for pi05_droid_trajectory_overlay.

This script runs the fine-tuned pi0.5 checkpoint that was trained to follow
trajectories drawn on the exterior camera image. At inference time it:

  1. Decomposes the instruction into manipulation steps (GPT)
  2. Detects objects and predicts a trajectory (Gemini + GPT)
  3. Draws the trajectory on the exterior image using the SAME TraceOverlayConfig
     defaults used during training (magenta, thickness=1, no outline)
  4. Sends the annotated image to the pi0.5 policy server
  5. Executes the returned actions on the robot
  6. Periodically re-plans and checks step completion

No eva framework dependency. Uses openpi_client + DROID RobotEnv directly.

Usage:
  # 1. Start the policy server (on GPU machine):
  uv run scripts/serve_policy.py policy:checkpoint \\
      --policy.config=pi05_droid_finetune \\
      --policy.dir=<path_to_trajectory_overlay_checkpoint>

  # 2. Run this script (on robot laptop):
  uv run examples/droid_trajectory_overlay/main.py \\
      --remote-host <policy_server_ip> \\
      --external-camera left \\
      --left-camera-id <id> --right-camera-id <id> --wrist-camera-id <id>

Requires:
  GEMINI_API_KEY and OPENAI_API_KEY environment variables.
"""

import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro

from traj_vis_utils import TraceOverlayConfig, add_trace_overlay
from trajectory_predictor import (
    encode_pil_image,
    query_step_completion,
    query_target_location,
    query_target_objects,
    query_trajectory,
    rescale_trajectory,
    resize_for_api,
)
from inference_visualizer import InferenceVisualizer, InferenceInfo

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Camera IDs (ZED serial numbers)
    left_camera_id: str = "<your_camera_id>"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "<your_camera_id>"

    # Which external camera to use for the policy
    external_camera: str | None = None  # "left" or "right"

    # Rollout parameters
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Trajectory prediction
    trajectory_source: str = "gpt"  # "gpt", "retrieval", or "fallback"
    gpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    plan_freq: int = 10  # Re-plan every N steps
    max_plan_count: int = 20  # Max replanning calls per trajectory
    check_interval: int = 20  # Check step completion every N steps since last plan

    # Retrieval+warp settings (only used when trajectory_source="retrieval" or "fallback")
    droid_5k_root: str = "/home/jianih/common-data/jiani_common/DROID_5K_processed"
    tether_root: str = "/home/asethi04/ROBOTICS/tether"

    # Output
    save_dir: str = "trajectory_overlay_runs"
    no_overlay: bool = False  # Disable trajectory overlay (for ablation)

    # Visualization
    show_display: bool = True  # Show live OpenCV debug window
    save_debug_frames: bool = True  # Save debug frames to disk
    display_width: int = 320  # Width of each panel in debug display
    save_frames_every: int = 5  # Save a debug frame every N steps

    # Instruction cache
    instruction_cache_path: str | None = None


# ---------------------------------------------------------------------------
# Ctrl+C protection during server calls
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _extract_observation(args: Args, obs_dict: dict) -> dict:
    """Extract images and robot state from DROID observation dict."""
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop alpha, convert BGR to RGB
    for name, img in [("left", left_image), ("right", right_image), ("wrist", wrist_image)]:
        if img is None:
            raise RuntimeError(f"Missing {name} camera image. Check camera IDs.")

    left_image = left_image[..., :3][..., ::-1]
    right_image = right_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }


def _draw_trajectory_on_image(
    image_rgb: np.ndarray,
    trajectory_points: list[list[float]],
    current_index: int = 0,
    config: TraceOverlayConfig | None = None,
) -> np.ndarray:
    """Draw trajectory overlay on image using training-matched config.

    During training, the overlay was drawn with current_index=t (the current
    frame), so the yellow dot moves along the trajectory as the episode
    progresses.  At inference time, we approximate this by tracking how many
    action steps have been executed since the trajectory was predicted.

    Args:
        image_rgb: uint8 RGB image (H, W, 3)
        trajectory_points: list of [x, y] pixel coordinates
        current_index: position of the yellow dot along the trajectory
        config: TraceOverlayConfig, defaults match training

    Returns:
        uint8 RGB image with trajectory overlay
    """
    if len(trajectory_points) < 2:
        return image_rgb

    pts = [tuple(p) for p in trajectory_points]
    idx = min(current_index, len(pts) - 1)
    # Use num_interpolated=100 to smooth sparse GPT waypoints into a
    # visually dense curve matching what the model saw during training
    # (training had ~160 dense SAM2 points with num_interpolated=0,
    # which is visually equivalent to fewer points + interpolation).
    result = add_trace_overlay(image_rgb, pts, current_index=idx, config=config,
                               num_interpolated=100)
    return np.array(result.convert("RGB"))


# ---------------------------------------------------------------------------
# Trajectory planning state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlanningState:
    """Mutable state for the trajectory planning loop."""
    steps: list[dict] = dataclasses.field(default_factory=list)
    step_idx: int = 0
    pred_traj: dict | None = None
    current_end_point: list[float] | None = None
    img_history: list[Image.Image] = dataclasses.field(default_factory=list)
    plan_count: int = 0
    steps_since_last_plan: int = 0
    full_task: str = ""  # the original complete instruction

    def reset_step(self):
        self.pred_traj = None
        self.current_end_point = None
        self.img_history = []
        self.steps_since_last_plan = 0

    @property
    def current_step(self) -> dict | None:
        if self.step_idx < len(self.steps):
            return self.steps[self.step_idx]
        return None

    @property
    def all_steps_done(self) -> bool:
        return self.step_idx >= len(self.steps)


def _enrich_step_context(planning: PlanningState) -> tuple[str, str, str]:
    """Pull target info from later steps when current step has empty target_location.

    For multi-step tasks like "Remove X from Y and put it on Z", step 0 has
    target_location="" but step 1 has target_location="on Z" and
    target_related_object="Z". We merge this context so the trajectory
    generator knows WHERE the object should ultimately go.

    Returns (target_location, target_related_object, extra_detect_query).
    """
    step_data = planning.current_step
    target_location = step_data["target_location"]
    target_related = step_data["target_related_object"]
    extra_query = ""

    if not target_location and planning.step_idx + 1 < len(planning.steps):
        next_step = planning.steps[planning.step_idx + 1]
        if next_step.get("target_location"):
            target_location = next_step["target_location"]
            print(f"[context] Enriched target_location from step {planning.step_idx + 1}: \"{target_location}\"")
        if next_step.get("target_related_object") and not target_related:
            target_related = next_step["target_related_object"]
            extra_query = target_related
            print(f"[context] Enriched target_related from step {planning.step_idx + 1}: \"{target_related}\"")

    return target_location, target_related, extra_query


def _predict_trajectory(
    planning: PlanningState,
    ext_image_rgb: np.ndarray,
    args: Args,
    target_location_point: list[float] | None = None,
) -> dict | None:
    """Run object detection + trajectory generation for the current step.

    Handles three key issues:
    1. Missing object detections — retries individually via query_target_location
    2. Empty target_location — enriches from subsequent steps in multi-step tasks
    3. Fuzzy object name matching — tolerates Gemini labeling differences

    Returns trajectory dict with points in original image coordinates, or None.
    """
    step_data = planning.current_step
    if step_data is None:
        return None

    manipulating_object = step_data["manipulating_object"]

    # Enrich context from later steps if current step has empty target
    target_location, target_related_object, extra_query = _enrich_step_context(planning)

    # Build query list — include extra objects from context enrichment
    detect_queries = [manipulating_object]
    if target_related_object:
        detect_queries.append(target_related_object)
    if extra_query and extra_query not in detect_queries:
        detect_queries.append(extra_query)
    detect_queries = list(set(q for q in detect_queries if q.strip()))

    # Resize for API
    pil_img = Image.fromarray(ext_image_rgb).convert("RGB")
    original_size = pil_img.size
    api_img = resize_for_api(pil_img, max_size=1024)
    api_size = api_img.size

    # Detect objects (with automatic retry for missing ones)
    object_locations = query_target_location(api_img, detect_queries, model_name=args.gemini_model)
    if object_locations is None:
        print("[trajectory] Object detection failed completely")
        return None

    manip_pt = object_locations.get(manipulating_object)
    target_pt = object_locations.get(target_related_object) if target_related_object else None

    # If manipulating object not found, use first available detection
    if manip_pt is None and object_locations:
        manip_pt = next(iter(object_locations.values()))
        print(f"[trajectory] Using fallback detection for '{manipulating_object}'")

    if manip_pt is None:
        print(f"[trajectory] Cannot find manipulating object '{manipulating_object}'")
        return None

    # If target object not found, use last available detection (if different from manip)
    if target_pt is None and target_related_object:
        remaining = {k: v for k, v in object_locations.items()
                     if v != manip_pt}
        if remaining:
            target_pt = next(iter(remaining.values()))
            print(f"[trajectory] Using fallback detection for '{target_related_object}'")

    # If still no target, set it to None — GPT will handle it via target_location text
    if target_pt is None:
        target_pt = manip_pt  # GPT will use target_location string to determine direction

    # Generate trajectory
    img_encoded = encode_pil_image(api_img)
    trajectory = query_trajectory(
        img=api_img,
        img_encoded=img_encoded,
        task=step_data["step"],
        manipulating_object=manipulating_object,
        manipulating_object_point=manip_pt,
        target_related_object=target_related_object or "target area",
        target_related_object_point=target_pt,
        target_location=target_location,
        gpt_model=args.gpt_model,
        target_location_point=target_location_point,
        full_task=planning.full_task,
    )

    if not trajectory or "trajectory" not in trajectory:
        print("[trajectory] Trajectory generation failed")
        return None

    # Store end point (in API image coords) for re-planning
    if planning.current_end_point is None and "end_point" in trajectory:
        planning.current_end_point = trajectory["end_point"]

    # Rescale to original image coordinates
    trajectory = rescale_trajectory(trajectory, api_size, original_size)
    return trajectory


def _check_step_completion(planning: PlanningState, args: Args) -> bool:
    """Check if current step is complete using Gemini."""
    if not planning.img_history:
        return False

    step_data = planning.current_step
    if step_data is None:
        return False

    result = query_step_completion(
        [planning.img_history[-1]],
        step=step_data["step"],
        model_name=args.gemini_model,
        max_images=1,
    )
    return result.get("is_complete", False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    assert args.external_camera in ("left", "right"), (
        f"--external-camera must be 'left' or 'right', got {args.external_camera}"
    )

    # Import DROID env
    from droid.robot_env import RobotEnv

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created DROID env")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"Connected to policy server at {args.remote_host}:{args.remote_port}")

    # TraceOverlayConfig with training defaults (red-to-pink gradient, yellow dot)
    overlay_config = TraceOverlayConfig()

    # Initialize trajectory source
    from trajectory_source import GPTTrajectorySource, RetrievalWarpTrajectorySource, FallbackTrajectorySource
    if args.trajectory_source == "gpt":
        traj_source = GPTTrajectorySource(gpt_model=args.gpt_model, gemini_model=args.gemini_model)
        print(f"Trajectory source: GPT ({args.gpt_model})")
    elif args.trajectory_source == "retrieval":
        traj_source = RetrievalWarpTrajectorySource(
            droid_5k_root=args.droid_5k_root, tether_root=args.tether_root, gemini_model=args.gemini_model,
        )
        print(f"Trajectory source: Retrieval+Warp (data: {args.droid_5k_root})")
    elif args.trajectory_source == "fallback":
        retrieval = RetrievalWarpTrajectorySource(
            droid_5k_root=args.droid_5k_root, tether_root=args.tether_root, gemini_model=args.gemini_model,
        )
        gpt = GPTTrajectorySource(gpt_model=args.gpt_model, gemini_model=args.gemini_model)
        traj_source = FallbackTrajectorySource(retrieval, gpt)
        print(f"Trajectory source: Fallback (retrieval → GPT)")
    else:
        raise ValueError(f"Unknown trajectory source: {args.trajectory_source}")

    # Visualizer
    visualizer = InferenceVisualizer(
        save_dir=os.path.join(args.save_dir, "viz") if args.save_debug_frames else None,
        display_width=args.display_width,
        show_window=args.show_display,
        save_every_n=args.save_frames_every,
    )
    viz_info = InferenceInfo()

    # Instruction cache
    instruction_cache = {}
    cache_path = args.instruction_cache_path or os.path.join(args.save_dir, "instruction_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            instruction_cache = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    while True:
        instruction = input("\nEnter instruction (or 'q' to quit): ").strip()
        if instruction.lower() == "q":
            break

        # --- Step 1: Decompose instruction ---
        planning = PlanningState(full_task=instruction)

        if instruction in instruction_cache:
            planning.steps = instruction_cache[instruction].get("steps", [])
            print(f"[cache hit] Reusing cached decomposition for: {instruction!r}")
        else:
            # Capture an image for context
            first_obs = _extract_observation(args, env.get_observation())
            first_img = first_obs[f"{args.external_camera}_image"]
            pil_first = resize_for_api(Image.fromarray(first_img).convert("RGB"))
            first_encoded = encode_pil_image(pil_first)

            target = query_target_objects(instruction, gpt_model=args.gpt_model, img_encoded=first_encoded)
            planning.steps = target.get("steps", [])
            instruction_cache[instruction] = target

        if not planning.steps:
            print("No steps extracted from instruction. Skipping.")
            continue

        print(f"\nDecomposed into {len(planning.steps)} step(s):")
        for i, s in enumerate(planning.steps):
            print(f"  {i + 1}. {s['step']} (manipulate: {s['manipulating_object']}, target: {s['target_location']})")

        current_instruction = planning.steps[0]["step"]

        # --- Step 2: Run rollout ---
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        run_dir = os.path.join(args.save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # Save instruction
        with open(os.path.join(run_dir, "instruction.txt"), "w") as f:
            f.write(instruction)

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        video_frames = []
        trajectory_log = []

        # Reset visualizer state
        viz_info = InferenceInfo(
            instruction=instruction,
            total_steps=len(planning.steps),
            max_plans=args.max_plan_count,
            open_loop_horizon=args.open_loop_horizon,
        )
        if planning.steps:
            viz_info.current_step = planning.steps[0]["step"]

        # Per-run debug dir
        viz_run_dir = os.path.join(run_dir, "viz") if args.save_debug_frames else None
        if viz_run_dir:
            visualizer.save_dir = viz_run_dir
            os.makedirs(viz_run_dir, exist_ok=True)
            for d in ["debug_frames", "raw_frames", "annotated_frames"]:
                os.makedirs(os.path.join(viz_run_dir, d), exist_ok=True)
            visualizer._frames_dir = os.path.join(viz_run_dir, "debug_frames")
            visualizer._raw_dir = os.path.join(viz_run_dir, "raw_frames")
            visualizer._annotated_dir = os.path.join(viz_run_dir, "annotated_frames")
        visualizer._frame_count = 0

        bar = tqdm.tqdm(range(args.max_timesteps), desc="Running rollout")
        print("Press Ctrl+C to stop early. Press 'q' in display window to stop.")

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation())
                ext_image = curr_obs[f"{args.external_camera}_image"]

                # --- Analysis / replanning ---
                should_plan = (
                    planning.plan_count < args.max_plan_count
                    and not planning.all_steps_done
                    and (t_step == 0 or (args.plan_freq > 0 and t_step % args.plan_freq == 0))
                )

                if should_plan:
                    if args.show_display:
                        visualizer.show_planning_status(
                            f"Planning step {planning.step_idx + 1}/{len(planning.steps)}: "
                            f"{planning.current_step['step'] if planning.current_step else '?'}"
                        )

                    pil_img = resize_for_api(Image.fromarray(ext_image).convert("RGB"))
                    planning.img_history.append(pil_img)

                    # Check step completion (skip on first plan)
                    if planning.pred_traj is not None and len(planning.img_history) > 1:
                        is_complete = _check_step_completion(planning, args)
                        if is_complete:
                            print(f"\n[step {planning.step_idx}] COMPLETE: {planning.current_step['step']}")
                            planning.step_idx += 1
                            planning.reset_step()

                            if planning.all_steps_done:
                                print("[all steps completed]")
                                break

                            current_instruction = planning.current_step["step"]
                            print(f"[advancing to step {planning.step_idx}]: {current_instruction}")

                            pil_img_new = resize_for_api(Image.fromarray(ext_image).convert("RGB"))
                            planning.img_history = [pil_img_new]

                    # Predict trajectory
                    if not planning.all_steps_done:
                        target_pt = planning.current_end_point if planning.pred_traj is not None else None
                        traj = _predict_trajectory(planning, ext_image, args, target_location_point=target_pt)
                        if traj is not None:
                            planning.pred_traj = traj
                            planning.plan_count += 1
                            planning.steps_since_last_plan = 0

                            bar.set_postfix(step=planning.step_idx, plans=planning.plan_count)

                            # Save trajectory image for debugging
                            traj_pts = [tuple(p) for p in traj["trajectory"]]
                            if len(traj_pts) >= 2:
                                traj_img = add_trace_overlay(ext_image, traj_pts, config=overlay_config)
                                traj_img.save(os.path.join(run_dir, f"traj_{t_step:04d}.jpg"))

                            trajectory_log.append({
                                "t_step": t_step,
                                "step_idx": planning.step_idx,
                                "trajectory": traj,
                            })
                else:
                    planning.steps_since_last_plan += 1

                # --- Draw trajectory overlay on exterior image ---
                # During training, current_index advanced 1 per frame across ~160
                # trajectory points.  At inference, the GPT trajectory has only
                # ~3-10 points but the robot runs ~160 steps between re-plans.
                # We map robot steps proportionally into trajectory point space
                # so the dot traverses the line at the same visual rate as training.
                annotated_ext_image = ext_image.copy()
                if not args.no_overlay and planning.pred_traj is not None:
                    traj_pts = planning.pred_traj.get("trajectory", [])
                    n_traj = len(traj_pts)
                    if n_traj >= 2 and args.plan_freq > 0:
                        # Map steps_since_last_plan ∈ [0, plan_freq) → [0, n_traj-1]
                        frac = planning.steps_since_last_plan / args.plan_freq
                        traj_current_idx = min(int(frac * (n_traj - 1)), n_traj - 1)
                    else:
                        traj_current_idx = 0
                    annotated_ext_image = _draw_trajectory_on_image(
                        ext_image, traj_pts, current_index=traj_current_idx, config=overlay_config,
                    )

                video_frames.append(ext_image)  # Save raw for video

                # --- Policy inference ---
                model_input_ext = None
                model_input_wrist = None
                inference_start = time.time()

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    model_input_ext = image_tools.resize_with_pad(annotated_ext_image, 224, 224)
                    model_input_wrist = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)

                    request_data = {
                        "observation/exterior_image_1_left": model_input_ext,
                        "observation/wrist_image_left": model_input_wrist,
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": current_instruction,
                    }

                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                inference_ms = (time.time() - inference_start) * 1000

                # Select action from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                action = np.clip(action, -1, 1)
                env.step(action)

                # --- Update visualization ---
                viz_info.t_step = t_step
                viz_info.step_idx = planning.step_idx
                viz_info.current_step = planning.current_step["step"] if planning.current_step else ""
                viz_info.plan_count = planning.plan_count
                viz_info.action_idx = actions_from_chunk_completed
                viz_info.trajectory_points = len(planning.pred_traj.get("trajectory", [])) if planning.pred_traj else 0
                viz_info.inference_time_ms = inference_ms

                key = visualizer.update(
                    raw_exterior=ext_image,
                    annotated_exterior=annotated_ext_image,
                    wrist_image=curr_obs["wrist_image"],
                    info=viz_info,
                    model_input_ext=model_input_ext,
                    model_input_wrist=model_input_wrist,
                )

                # Check for quit key
                if key == ord("q"):
                    print("\nQuit requested via display window.")
                    break

                # Match DROID control frequency
                elapsed = time.time() - start_time
                if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)

            except KeyboardInterrupt:
                print("\nRollout interrupted by user.")
                break

        # --- Save visualizer summary ---
        visualizer.save_summary(viz_info)

        # --- Save results ---
        if video_frames:
            try:
                from moviepy.editor import ImageSequenceClip
                video = np.stack(video_frames)
                video_path = os.path.join(run_dir, "rollout.mp4")
                ImageSequenceClip(list(video), fps=10).write_videofile(video_path, codec="libx264", logger=None)
                print(f"Video saved to {video_path}")
            except ImportError:
                print("moviepy not installed, skipping video save")

        if trajectory_log:
            with open(os.path.join(run_dir, "trajectories.json"), "w") as f:
                json.dump(trajectory_log, f, indent=2)

        # Log result
        success_input = input("Did the rollout succeed? (y/n): ").strip().lower()
        with open(os.path.join(run_dir, "result.txt"), "w") as f:
            f.write(f"success: {success_input}\n")
            f.write(f"instruction: {instruction}\n")
            f.write(f"steps_completed: {planning.step_idx}/{len(planning.steps)}\n")
            f.write(f"total_timesteps: {t_step + 1}\n")
            f.write(f"total_plans: {planning.plan_count}\n")

        # Save instruction cache
        with open(cache_path, "w") as f:
            json.dump(instruction_cache, f, indent=2)

        if input("Run another? (y/n): ").strip().lower() != "y":
            break
        env.reset()

    visualizer.close()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
