"""
Minimal DROID inference client for the BASE pi0.5 DROID checkpoint.

No LLM calls, no trajectory overlay, no instruction decomposition. The user
instruction is passed verbatim as the `prompt` field in the policy request,
and the policy decides what to do with it.

Drop-in for main.py when you want to test pi0.5-droid without any of the
trajectory-guidance modifications.

Usage:
    python -u main_pi05.py \
        --remote-host 127.0.0.1 --remote-port 8000 \
        --external-camera left \
        --left-camera-id 26368109 \
        --right-camera-id 25455306 \
        --wrist-camera-id 15512737 \
        --save-dir ~/pi-trajectory-overlay/runs_pi05

Kill from another terminal:
    tmux kill-session -t rollout 2>/dev/null; pkill -9 -f main_pi05.py
"""

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro

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

    # Rollout
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Output
    save_dir: str = "runs_pi05"

    # Debug: dump every step's full-res + 224x224 (ext + wrist) to <run_dir>/frames/.
    # Useful for inspecting exactly what the policy is seeing. ~175 KB/step,
    # ~100 MB for a 600-step rollout. Off by default — enable with --save-frames.
    save_frames: bool = False


# ---------------------------------------------------------------------------
# Ctrl+C protection while the policy server call is in flight
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
# Observation extraction — identical to main.py: BGR→RGB + vertical flip on
# the external camera (the two ZED 2s are mounted upside-down on this rig).
# Wrist ZED-M is NOT flipped.
# ---------------------------------------------------------------------------

def _extract_observation(args: Args, obs_dict: dict) -> dict:
    image_observations = obs_dict.get("image", {})
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    required = [("wrist", wrist_image)]
    if args.external_camera == "left":
        required.append(("left", left_image))
    else:
        required.append(("right", right_image))

    for name, img in required:
        if img is None:
            available = list(image_observations.keys())
            raise RuntimeError(
                f"Missing {name} camera image.\n"
                f"  args.left_camera_id={args.left_camera_id!r}\n"
                f"  args.right_camera_id={args.right_camera_id!r}\n"
                f"  args.wrist_camera_id={args.wrist_camera_id!r}\n"
                f"  args.external_camera={args.external_camera!r}\n"
                f"  Available image keys: {available}"
            )

    def _to_rgb(img):
        if img is None:
            return None
        return img[..., :3][..., ::-1]

    left_image = _to_rgb(left_image)
    right_image = _to_rgb(right_image)
    wrist_image = _to_rgb(wrist_image)

    # External cameras are mounted 180° rotated on this Franka rig (upside-down
    # AND mirrored), so apply a full 180° rotation — flip both axes. The wrist
    # ZED-M is mounted correctly and is left alone.
    if left_image is not None:
        left_image = np.ascontiguousarray(left_image[::-1, ::-1])
    if right_image is not None:
        right_image = np.ascontiguousarray(right_image[::-1, ::-1])

    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    assert args.external_camera in ("left", "right"), (
        f"--external-camera must be 'left' or 'right', got {args.external_camera}"
    )

    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created DROID env")

    # Camera warmup — same logic as main.py.
    print("Warming up cameras (up to 60s)...")
    max_attempts = 120
    required_ext_id = args.left_camera_id if args.external_camera == "left" else args.right_camera_id
    for attempt in range(max_attempts):
        obs = env.get_observation()
        img_dict = obs.get("image", {})
        keys = list(img_dict.keys()) if isinstance(img_dict, dict) else []
        have_ext = any(required_ext_id in k and "left" in k for k in keys)
        have_wrist = any(args.wrist_camera_id in k and "left" in k for k in keys)
        if have_ext and have_wrist:
            print(f"  Cameras warmed up after {attempt + 1} attempts; keys: {keys}")
            break
        if attempt % 10 == 0 or attempt < 5:
            print(f"  [warmup {attempt + 1}/{max_attempts}] keys: {keys}; "
                  f"have_{args.external_camera}={have_ext} have_wrist={have_wrist}")
        time.sleep(0.5)
    else:
        print(f"WARNING: cameras did not fully warm up in {max_attempts * 0.5}s; continuing")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"Connected to policy server at {args.remote_host}:{args.remote_port}")

    os.makedirs(args.save_dir, exist_ok=True)

    while True:
        instruction = input("\nEnter instruction (or 'q' to quit): ").strip()
        if instruction.lower() == "q":
            break
        if not instruction:
            continue

        # Per-run directory. `frames/` (optional, --save-frames) holds every
        # image we process, written incrementally so a hard kill (pkill -9)
        # still leaves the full visual record on disk.
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        run_dir = os.path.join(args.save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        frames_dir = os.path.join(run_dir, "frames")
        if args.save_frames:
            os.makedirs(frames_dir, exist_ok=True)
        with open(os.path.join(run_dir, "instruction.txt"), "w") as f:
            f.write(instruction)
        action_log_path = os.path.join(run_dir, "actions.log")

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        video_frames = []

        bar = tqdm.tqdm(range(args.max_timesteps), desc="Running rollout")
        print("Press Ctrl+C to stop early.")
        if args.save_frames:
            print(f"Per-step frames → {frames_dir}")

        t_step = 0
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation())
                ext_image = curr_obs[f"{args.external_camera}_image"]
                wrist_image = curr_obs["wrist_image"]
                video_frames.append(ext_image)

                # Compute 224x224 model inputs up front so we can dump them to
                # disk every step even if we don't query the policy this step.
                # These are the EXACT tensors sent to the policy server.
                model_input_ext = image_tools.resize_with_pad(ext_image, 224, 224)
                model_input_wrist = image_tools.resize_with_pad(wrist_image, 224, 224)

                # --- Optionally save every frame to disk, incrementally ---
                # Each .save() call flushes to the OS, so a SIGKILL leaves all
                # completed-step frames on disk. ~175 KB/step total at HD720.
                # Controlled by --save-frames; off by default.
                if args.save_frames:
                    Image.fromarray(ext_image).save(
                        os.path.join(frames_dir, f"{t_step:04d}_ext.jpg"), quality=85,
                    )
                    Image.fromarray(wrist_image).save(
                        os.path.join(frames_dir, f"{t_step:04d}_wrist.jpg"), quality=85,
                    )
                    Image.fromarray(model_input_ext).save(
                        os.path.join(frames_dir, f"{t_step:04d}_ext_224.jpg"), quality=90,
                    )
                    Image.fromarray(model_input_wrist).save(
                        os.path.join(frames_dir, f"{t_step:04d}_wrist_224.jpg"), quality=90,
                    )

                queried_this_step = False
                inference_ms = 0.0
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    queried_this_step = True
                    request_data = {
                        "observation/exterior_image_1_left": model_input_ext,
                        "observation/wrist_image_left": model_input_wrist,
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }
                    inference_start = time.time()
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    inference_ms = (time.time() - inference_start) * 1000

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                action = np.clip(action, -1, 1)

                env.step(action)

                with open(action_log_path, "a") as alf:
                    alf.write(
                        f"t={t_step:04d} | "
                        f"queried={int(queried_this_step)} | "
                        f"action={np.round(action, 3).tolist()} | "
                        f"gripper={action[-1]:.2f} | "
                        f"chunk_idx={actions_from_chunk_completed}/{args.open_loop_horizon} | "
                        f"inf_ms={inference_ms:.0f}\n"
                    )

                elapsed = time.time() - start_time
                if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)

            except KeyboardInterrupt:
                print("\nRollout interrupted by user.")
                break

        # Save video
        if video_frames:
            try:
                from moviepy.editor import ImageSequenceClip
                video = np.stack(video_frames)
                video_path = os.path.join(run_dir, "rollout.mp4")
                ImageSequenceClip(list(video), fps=10).write_videofile(
                    video_path, codec="libx264", logger=None,
                )
                print(f"Video saved to {video_path}")
            except ImportError:
                print("moviepy not installed, skipping video save")

        success = input("Did the rollout succeed? (y/n): ").strip().lower()
        with open(os.path.join(run_dir, "result.txt"), "w") as f:
            f.write(f"success: {success}\n")
            f.write(f"instruction: {instruction}\n")
            f.write(f"total_timesteps: {t_step + 1}\n")
            f.write(f"policy: pi05_droid (base, no trajectory overlay)\n")

        if input("Run another? (y/n): ").strip().lower() != "y":
            break
        env.reset()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
