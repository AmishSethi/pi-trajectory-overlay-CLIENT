"""
Minimal RoboLab inference client for the BASE pi0.5 DROID checkpoint.

No LLM calls, no trajectory overlay, no instruction decomposition. The task's
registered instruction (or a user-typed override) is passed verbatim as the
`prompt` field in the policy request, and the policy decides what to do.

Twin of `../main_pi05.py`: same inline rollout loop, same interactive
while-True instruction prompt, same per-rollout artifact layout. The only
forced differences versus the real-robot client are:

    - argparse + Isaac Sim AppLauncher bootstrap (AppLauncher takes an
      argparse.ArgumentParser, not tyro).
    - _extract_observation reads the RoboLab sim obs dict (torch tensors)
      instead of the DROID websocket dict.
    - _integrate_velocity turns the server's normalized joint-velocity
      action (pi05_droid action space) into a joint-position target, because
      RoboLab's env.step expects joint-position commands. Math below mirrors
      RoboLab's own Pi0DroidJointvelClient byte-for-byte (see
      ../third_party/RoboLab/robolab/inference/pi0_jointvel.py, commit
      684d3f2 — the commit that introduced pi05_jointvel in RoboLab).
    - env.step takes a torch tensor on env.device.

Usage:
    python -u main_robolab.py \
        --task BananaOnPlateTask \
        --remote-host 127.0.0.1 --remote-port 8001 \
        --save-dir ~/runs_robolab \
        --headless

    # Smoke test without a server:
    python -u main_robolab.py --task BananaOnPlateTask --fake-policy \
        --max-timesteps 3 --headless

Kill from another terminal:
    tmux kill-session -t rollout 2>/dev/null; pkill -9 -f main_robolab.py
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

faulthandler.enable()

SIM_CONTROL_FREQUENCY = 15

# Franka DROID joint velocity limits (rad/s), mirroring RoboLab's
# Pi0DroidJointvelClient defaults. pi05_droid emits v_norm in [-1, 1]; we
# multiply by these per-joint limits to recover rad/s before integration.
DROID_VEL_LIMITS = np.asarray(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32,
)


@dataclasses.dataclass
class Args:
    # Task selection
    task: str = ""  # required — RoboLab registered task class name, e.g. "BananaOnPlateTask"
    instruction_type: str = "default"
    task_dirs: tuple[str, ...] = ("benchmark",)

    # Rollout
    max_timesteps: int = 600
    open_loop_horizon: int = 8
    num_envs: int = 1

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    fake_policy: bool = False

    # Output
    save_dir: str = "runs_robolab"
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
# Observation extraction — sim version. Same output-dict shape as
# ../main_pi05.py._extract_observation, so downstream code can stay identical.
# ---------------------------------------------------------------------------

def _extract_observation(args: "Args", obs_dict: dict, env_id: int = 0) -> dict:
    image_obs = obs_dict["image_obs"]
    proprio = obs_dict["proprio_obs"]

    def _to_numpy_image(t):
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        arr = np.asarray(t)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def _to_numpy_vec(t):
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    external_image = _to_numpy_image(image_obs["external_cam"][env_id])
    wrist_image = _to_numpy_image(image_obs["wrist_cam"][env_id])
    joint_position = _to_numpy_vec(proprio["arm_joint_pos"][env_id])
    gripper_position = _to_numpy_vec(proprio["gripper_pos"][env_id])

    return {
        "external_image": external_image,
        "wrist_image": wrist_image,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


# ---------------------------------------------------------------------------
# Velocity → position integration. Mirrors the math in RoboLab's
# Pi0DroidJointvelClient (commit 684d3f2):
#
#     q_target[t+1] = q_target[t] + clip(v_norm, -1, 1) * vel_limits * dt
#
# q_target is re-anchored to the measured arm pose at every new policy
# chunk, so open-loop integration error stays bounded to one chunk.
# ---------------------------------------------------------------------------

def _integrate_velocity(v_norm: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    v_clipped = np.clip(np.asarray(v_norm, dtype=np.float32), -1.0, 1.0)
    return q_target + v_clipped * DROID_VEL_LIMITS * (1.0 / SIM_CONTROL_FREQUENCY)


# ---------------------------------------------------------------------------
# Fake policy (used when --fake-policy is set; no server required)
# ---------------------------------------------------------------------------

class FakePolicy:
    """Returns zero-velocity action chunks. Drop-in replacement for
    websocket_client_policy.WebsocketClientPolicy.infer(), useful to smoke-
    test the sim side when you don't want to spin up a real policy server."""

    def __init__(self, action_dim: int = 8, chunk_length: int = 16) -> None:
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.last_request: dict | None = None
        self.infer_calls = 0

    def infer(self, request: dict) -> dict:
        self.last_request = request
        self.infer_calls += 1
        return {"actions": np.zeros((self.chunk_length, self.action_dim), dtype=np.float32)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    assert args.task, "--task is required (e.g. BananaOnPlateTask)"

    # Isaac Sim must already be booted by __main__ before we hit these imports.
    from robolab.core.environments.runtime import create_env
    from robolab.core.logging.recorder_manager import patch_recorder_manager
    from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs
    import torch

    patch_recorder_manager()
    auto_register_droid_envs(task_dirs=list(args.task_dirs), task=args.task)
    env, env_cfg = create_env(
        args.task,
        num_envs=args.num_envs,
        use_fabric=True,
        instruction_type=args.instruction_type,
    )
    print(f"Created RoboLab env for task {args.task}")

    if args.fake_policy:
        policy_client = FakePolicy()
        print("[main_robolab] --fake-policy: using FakePolicy (no server required).")
    else:
        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        print(f"Connected to policy server at {args.remote_host}:{args.remote_port}")

    os.makedirs(args.save_dir, exist_ok=True)
    default_instruction = env_cfg.instruction

    while True:
        try:
            instruction = input(
                f"\nEnter instruction (empty = task default '{default_instruction}', 'q' to quit): "
            ).strip()
        except EOFError:
            break
        if instruction.lower() == "q":
            break
        if not instruction:
            instruction = default_instruction

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
        q_target = None  # re-anchored each new policy chunk

        # RoboLab examples reset twice before a rollout to let physics settle.
        obs, _ = env.reset()
        obs, _ = env.reset()

        auto_success = None  # set from env term/trunc if the episode ends early
        bar = tqdm.tqdm(range(args.max_timesteps), desc="Running rollout")
        print("Press Ctrl+C to stop early.")
        if args.save_frames:
            print(f"Per-step frames → {frames_dir}")

        t_step = 0
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, obs)
                ext_image = curr_obs["external_image"]
                wrist_image = curr_obs["wrist_image"]
                video_frames.append(ext_image)

                # Compute 224x224 model inputs up front so we can dump them to
                # disk every step even if we don't query the policy this step.
                # These are the EXACT tensors sent to the policy server.
                model_input_ext = image_tools.resize_with_pad(ext_image, 224, 224)
                model_input_wrist = image_tools.resize_with_pad(wrist_image, 224, 224)

                # --- Optionally save every frame to disk, incrementally ---
                # Each .save() call flushes to the OS, so a SIGKILL leaves all
                # completed-step frames on disk.
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
                    # Re-anchor velocity integrator to the CURRENTLY measured
                    # joint position; bounds open-loop drift to one chunk.
                    q_target = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Velocity → position integration (sim env expects joint-
                # position targets; the server emits joint velocities).
                q_target = _integrate_velocity(action[:7], q_target)

                # Binarize gripper (same threshold as main_pi05.py)
                if action[-1].item() > 0.5:
                    gripper = np.ones((1,), dtype=np.float32)
                else:
                    gripper = np.zeros((1,), dtype=np.float32)

                action = np.concatenate([q_target, gripper])

                # Sim takes a torch tensor on env.device, shape [num_envs, 8].
                action_t = torch.from_numpy(action).to(env.device).unsqueeze(0)
                if env.num_envs > 1:
                    action_t = action_t.expand(env.num_envs, -1).contiguous()

                obs, _, term, trunc, _ = env.step(action_t)

                with open(action_log_path, "a") as alf:
                    alf.write(
                        f"t={t_step:04d} | "
                        f"queried={int(queried_this_step)} | "
                        f"action={np.round(action, 3).tolist()} | "
                        f"gripper={action[-1]:.2f} | "
                        f"chunk_idx={actions_from_chunk_completed}/{args.open_loop_horizon} | "
                        f"inf_ms={inference_ms:.0f}\n"
                    )

                # Sim can terminate episodes early on task success / truncation.
                if bool(term[0].item()):
                    auto_success = True
                    print(f"[main_robolab] Task terminated (success) at step {t_step + 1}.")
                    break
                if bool(trunc[0].item()):
                    auto_success = False
                    print(f"[main_robolab] Episode truncated at step {t_step + 1}.")
                    break

                elapsed = time.time() - start_time
                if elapsed < 1 / SIM_CONTROL_FREQUENCY:
                    time.sleep(1 / SIM_CONTROL_FREQUENCY - elapsed)

            except KeyboardInterrupt:
                print("\nRollout interrupted by user.")
                break

        # Save video
        if video_frames:
            try:
                import imageio.v2 as imageio
                video_path = os.path.join(run_dir, "rollout.mp4")
                imageio.mimwrite(
                    video_path, video_frames, fps=15, codec="libx264", quality=8,
                )
                print(f"Video saved to {video_path}")
            except Exception as e:
                print(f"Could not save video: {e}")

        if auto_success is not None:
            print(f"[main_robolab] auto-detected success={auto_success}")
        try:
            success = input("Did the rollout succeed? (y/n): ").strip().lower()
        except EOFError:
            success = "auto" if auto_success is None else ("y" if auto_success else "n")

        with open(os.path.join(run_dir, "result.txt"), "w") as f:
            f.write(f"success: {success}\n")
            f.write(f"auto_success: {auto_success}\n")
            f.write(f"instruction: {instruction}\n")
            f.write(f"total_timesteps: {t_step + 1}\n")
            f.write(f"policy: pi05_droid (base, via {'fake' if args.fake_policy else 'server'})\n")

        try:
            again = input("Run another? (y/n): ").strip().lower()
        except EOFError:
            again = "n"
        if again != "y":
            break
        env.reset()


if __name__ == "__main__":
    # Isaac Sim MUST boot before any isaaclab / robolab import. Match the
    # dance that RoboLab's own run_eval.py does so we inherit its env-
    # discovery path.
    import argparse
    import cv2  # noqa: F401  (must be imported before isaaclab)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default")
    parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=["benchmark"])
    parser.add_argument("--max-timesteps", "--max_timesteps", type=int, default=600)
    parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=8)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--remote-host", "--remote_host", type=str, default="0.0.0.0")
    parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
    parser.add_argument("--fake-policy", "--fake_policy", action="store_true")
    parser.add_argument("--save-dir", "--save_dir", type=str, default="runs_robolab")
    parser.add_argument("--save-frames", "--save_frames", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    ns, _ = parser.parse_known_args()
    ns.enable_cameras = True

    _APP_LAUNCHER = AppLauncher(ns)
    _SIM_APP = _APP_LAUNCHER.app

    try:
        main(Args(
            task=ns.task,
            instruction_type=ns.instruction_type,
            task_dirs=tuple(ns.task_dirs),
            max_timesteps=ns.max_timesteps,
            open_loop_horizon=ns.open_loop_horizon,
            num_envs=ns.num_envs,
            remote_host=ns.remote_host,
            remote_port=ns.remote_port,
            fake_policy=ns.fake_policy,
            save_dir=ns.save_dir,
            save_frames=ns.save_frames,
        ))
    finally:
        _SIM_APP.close()
