"""
Real-robot MPC-overlay inference client for base pi0.5 DROID.

Combines the minimal policy loop of main_pi05.py with the client-side MPC
(CEM) refinement we developed in the simulator (simulator/main_robolab.py),
so the frozen VLA stays byte-identical to the baseline and we only touch
the sampled action chunk before sending it to the robot.

The MPC takes user-drawn 2D pixel waypoints (in the *flipped* ZED frame —
same frame the human sees on the laptop display and that main.py's overlay
client uses), projects them through the exterior-camera intrinsics and
extrinsics, and scores CEM candidates by a composite cost:
    prior (stay close to VLA) + arrow (masked 2D L2) +
    joint-limit + action-box (constraints) + smoothness +
    progress reward (MPCC-style: reward advancing along the arrow's arc length)

Everything else (gripper, VLA inference path, image flipping) matches
main_pi05.py so A/B with --guidance-mode=off is byte-identical to the
baseline rollout.

Waypoint source:
  • --canned-waypoints-json <path>      hand-specified {"waypoints_px": [[u,v]...]}
  • (fallback) static identity arrow    useful for smoke tests
  • --llm-plan-freq N >0                re-plan every N steps via Gemini+GPT
                                         (reuses trajectory_predictor.query_target_location
                                         + query_trajectory; same pipeline as main.py)

Usage:
    python -u main_mpc.py \\
        --remote-host 127.0.0.1 --remote-port 8000 \\
        --external-camera left \\
        --left-camera-id 26368109 --right-camera-id 25455306 --wrist-camera-id 15512737 \\
        --canned-waypoints-json ~/waypoints/banana_in_bowl.json \\
        --guidance-mode mpc \\
        --save-dir ~/pi-trajectory-overlay/runs_mpc

Kill from another terminal:
    tmux kill-session -t rollout 2>/dev/null; pkill -9 -f main_mpc.py
"""

import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import time
from typing import Literal

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


# ---------------------------------------------------------------------------
# Exterior-ZED calibration (same constants main.py uses).
#   - HD720 factory intrinsics for ZED serial 26368109 LEFT
#   - Extrinsics loaded from /home/asethi04/ROBOTICS/eva_pal_share/eva/utils/calibration.json
#     with a hard-coded fallback that matches the repo's calibration snapshot.
#   - image_flipped_180 = True because externals are mounted upside-down on this rig.
#   - ee_offset_from_flange = 0.0 — real-robot FK matches the existing
#     trajectory_cost tests (flange), unlike the sim which adds the 0.1034 m
#     panda_hand_tcp offset.
# ---------------------------------------------------------------------------
_REAL_K_INTRINSICS = np.array(
    [[532.66, 0.0, 641.305],
     [0.0, 532.55, 347.186],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_REAL_EXTRINSIC_FALLBACK = np.array(
    [0.09153819431911293, -0.01455634604254126, 0.2287419968377015,
     -1.5253417996893945, 0.019538164296171168, -1.55553965929186],
    dtype=np.float32,
)

_CALIBRATION_JSON_PATH = "/home/asethi04/ROBOTICS/eva_pal_share/eva/utils/calibration.json"

# Franka Panda joint velocity limits (rad/s), from the FCI datasheet.
_REAL_JOINT_VEL_SCALE = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
    dtype=np.float32,
)

# image HxW matches ZED HD720.
_REAL_IMAGE_HW = (720, 1280)

# Real robot uses flange FK (matches ZED calibration).
_REAL_EE_OFFSET_FROM_FLANGE = 0.0


def _load_extrinsics() -> np.ndarray:
    try:
        with open(_CALIBRATION_JSON_PATH) as f:
            calib = json.load(f)
        return np.asarray(calib["26368109_left"]["extrinsics"], dtype=np.float32)
    except (OSError, KeyError, json.JSONDecodeError) as e:
        print(f"[mpc] calibration.json missing/invalid ({e!r}); using fallback extrinsics.")
        return _REAL_EXTRINSIC_FALLBACK.copy()


_CACHED_EXTRINSIC = _load_extrinsics()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # Camera IDs (ZED serial numbers)
    left_camera_id: str = "<your_camera_id>"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "<your_camera_id>"
    external_camera: str | None = None  # "left" or "right"

    # Rollout
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Output
    save_dir: str = "runs_mpc"
    save_frames: bool = False

    # --- Guidance -----------------------------------------------------------
    # "off"  — baseline pi0.5 (byte-identical to main_pi05.py)
    # "mpc"  — client-side CEM refines the VLA chunk against user waypoints
    guidance_mode: Literal["off", "mpc"] = "off"

    # Waypoint source (only read when guidance_mode == "mpc"):
    #   If set, load {"waypoints_px": [[u,v], ...]} from this file (FLIPPED
    #   1280x720 frame — same frame you draw on, same frame main.py's overlay
    #   pipeline uses).
    canned_waypoints_json: str = ""

    # Optional LLM planning (matches main.py's cadence).
    # 0 disables; >0 re-plans every N steps via Gemini-ER + GPT-4o-mini.
    llm_plan_freq: int = 0
    gpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-robotics-er-1.5-preview"

    # --- MPC weights (same defaults as simulator balanced config) ----------
    mpc_lam_p: float = 1.0
    mpc_lam_a: float = 10.0
    mpc_lam_c: float = 100.0
    mpc_lam_s: float = 0.01
    mpc_lam_prog: float = 1.0   # MPCC-style progress reward (clamp-≥0 mean over chunk)

    # --- CEM hyperparameters ---------------------------------------------
    mpc_n_samples: int = 200
    mpc_n_iterations: int = 4
    mpc_n_elites: int = 20
    mpc_init_std: float = 0.05

    # MPC compute device. "auto" picks first CUDA if available, else "cpu".
    mpc_device: str = "auto"

    # Sliding-window arrow lookahead. None = legacy full-arrow target.
    # Typical 0.10–0.25 (fraction of arc length per chunk).
    mpc_arrow_lookahead: float | None = 0.15

    # Gripper controls
    mpc_freeze_gripper: bool = True
    mpc_gripper_reward_weight: float = 0.0
    mpc_gripper_zone_frac: float = 0.15
    mpc_gripper_force_override: bool = False


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
# Observation extraction — IDENTICAL to main_pi05.py
# (BGR→RGB + vertical+horizontal flip on exterior, wrist left alone).
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
                f"  args.external_camera={args.external_camera!r}\n"
                f"  Available image keys: {available}"
            )

    def _to_rgb(img):
        return None if img is None else img[..., :3][..., ::-1]

    left_image = _to_rgb(left_image)
    right_image = _to_rgb(right_image)
    wrist_image = _to_rgb(wrist_image)

    # Exterior ZEDs are mounted 180° rotated on this rig — flip both axes.
    if left_image is not None:
        left_image = np.ascontiguousarray(left_image[::-1, ::-1])
    if right_image is not None:
        right_image = np.ascontiguousarray(right_image[::-1, ::-1])

    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "joint_position": np.array(robot_state["joint_positions"], dtype=np.float32),
        "gripper_position": np.array([robot_state["gripper_position"]], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# MPC helpers: build GuidanceSpec, run CEM, return refined chunk.
# ---------------------------------------------------------------------------
def _load_canned_waypoints(path: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    wp = np.asarray(data["waypoints_px"], dtype=np.float32)
    if wp.ndim != 2 or wp.shape[-1] != 2 or wp.shape[0] < 2:
        raise ValueError(f"canned waypoints must be (K>=2, 2), got shape {wp.shape}")
    return wp


def _save_spec_snapshot(run_dir: str) -> None:
    """One-shot JSON containing intrinsics/extrinsics/joint_vel_scale so the
    post-hoc annotator (tmp/overnight_v2/scripts/annotate_with_fk.py) can
    reconstruct the same FK projection the MPC used."""
    with open(os.path.join(run_dir, "guidance_spec_snapshot.json"), "w") as f:
        json.dump({
            "K_intrinsics": _REAL_K_INTRINSICS.tolist(),
            "extrinsic_cam_in_base": _CACHED_EXTRINSIC.tolist(),
            "joint_vel_scale": _REAL_JOINT_VEL_SCALE.tolist(),
            "image_hw": list(_REAL_IMAGE_HW),
            "control_dt": 1.0 / DROID_CONTROL_FREQUENCY,
            "image_flipped_180": True,
            "ee_offset_from_flange": _REAL_EE_OFFSET_FROM_FLANGE,
        }, f, indent=2)


def _resolve_mpc_device(requested: str):
    import torch
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _refine_chunk_with_mpc(
    args: Args,
    pred_action_chunk: np.ndarray,  # (T, 8)
    q0_7: np.ndarray,                # (7,)
    waypoints_px: np.ndarray,        # (K, 2) in the FLIPPED 1280x720 frame
    device,
) -> np.ndarray:
    import torch
    from mpc_overlay import CEMParams, GuidanceSpec, MPCWeights, mpc_overlay

    K = torch.tensor(_REAL_K_INTRINSICS, dtype=torch.float32)
    ext = torch.tensor(_CACHED_EXTRINSIC, dtype=torch.float32)
    jvs = torch.tensor(_REAL_JOINT_VEL_SCALE, dtype=torch.float32)
    wp = torch.as_tensor(np.asarray(waypoints_px), dtype=torch.float32)
    q0 = torch.as_tensor(np.asarray(q0_7)[:7], dtype=torch.float32).unsqueeze(0)
    spec = GuidanceSpec(
        waypoints_px=wp,
        image_hw=_REAL_IMAGE_HW,
        K_intrinsics=K,
        extrinsic_cam_in_base=ext,
        image_flipped_180=True,
        control_dt=1.0 / DROID_CONTROL_FREQUENCY,
        joint_vel_scale=jvs,
        q0=q0,
        ee_offset_from_flange=_REAL_EE_OFFSET_FROM_FLANGE,
        arrow_lookahead=args.mpc_arrow_lookahead,
        gripper_reward_weight=float(args.mpc_gripper_reward_weight),
        gripper_zone_frac=float(args.mpc_gripper_zone_frac),
        gripper_force_override=bool(args.mpc_gripper_force_override),
    )
    weights = MPCWeights(
        lam_p=float(args.mpc_lam_p),
        lam_a=float(args.mpc_lam_a),
        lam_c=float(args.mpc_lam_c),
        lam_s=float(args.mpc_lam_s),
        lam_prog=float(args.mpc_lam_prog),
    )
    cem_params = CEMParams(
        n_samples=int(args.mpc_n_samples),
        n_iterations=int(args.mpc_n_iterations),
        n_elites=int(args.mpc_n_elites),
        init_std=float(args.mpc_init_std),
        freeze_gripper=bool(args.mpc_freeze_gripper),
    )
    a_vla = torch.as_tensor(np.asarray(pred_action_chunk).copy(), dtype=torch.float32, device=device)
    a_opt = mpc_overlay(a_vla, spec, weights, cem_params)
    return a_opt.detach().cpu().numpy().astype(pred_action_chunk.dtype)


# ---------------------------------------------------------------------------
# LLM planning subprocess wrapper (optional). Uses the same Gemini+GPT flow
# main.py uses — imported lazily so non-LLM runs don't need the API keys.
# ---------------------------------------------------------------------------
def _llm_plan_waypoints(args: Args, ext_image: np.ndarray, instruction: str):
    from trajectory_predictor import (
        encode_pil_image, query_target_location, query_trajectory, resize_for_api,
    )
    pil = Image.fromarray(ext_image).convert("RGB")
    resized, scale_x, scale_y = resize_for_api(pil)
    b64 = encode_pil_image(resized)
    target = query_target_location(b64, instruction, model=args.gemini_model)
    if target is None:
        return None
    traj = query_trajectory(b64, instruction, target, model=args.gpt_model)
    if traj is None or len(traj) < 2:
        return None
    wp_api = np.asarray(traj, dtype=np.float32)
    # resize_for_api shrinks to the API's preferred max dim; scale back to 1280x720.
    wp = np.stack([wp_api[:, 0] * scale_x, wp_api[:, 1] * scale_y], axis=-1)
    return wp.astype(np.float32)


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

    # Camera warmup — same logic as main_pi05.py.
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
            print(f"  [warmup {attempt + 1}/{max_attempts}] keys: {keys}")
        time.sleep(0.5)
    else:
        print(f"WARNING: cameras did not fully warm up in {max_attempts * 0.5}s; continuing")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"Connected to policy server at {args.remote_host}:{args.remote_port}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Resolve MPC device early (skips import if guidance is off).
    mpc_device = None
    if args.guidance_mode == "mpc":
        mpc_device = _resolve_mpc_device(args.mpc_device)
        print(f"[mpc] device={mpc_device}  "
              f"weights(p={args.mpc_lam_p}, a={args.mpc_lam_a}, c={args.mpc_lam_c}, "
              f"s={args.mpc_lam_s}, prog={args.mpc_lam_prog})  "
              f"CEM(N={args.mpc_n_samples}, I={args.mpc_n_iterations}, K={args.mpc_n_elites}, "
              f"std={args.mpc_init_std})  lookahead={args.mpc_arrow_lookahead}  "
              f"freeze_gripper={args.mpc_freeze_gripper}  "
              f"gripper_force_override={args.mpc_gripper_force_override}")

    while True:
        instruction = input("\nEnter instruction (or 'q' to quit): ").strip()
        if instruction.lower() == "q":
            break
        if not instruction:
            continue

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

        # Waypoint loading: canned JSON > LLM > error (in mpc mode).
        mpc_waypoints = None
        plan_count = 0
        using_canned_waypoints = False
        if args.guidance_mode == "mpc":
            _save_spec_snapshot(run_dir)
            if args.canned_waypoints_json:
                mpc_waypoints = _load_canned_waypoints(args.canned_waypoints_json)
                using_canned_waypoints = True
                plan_count = 1
                np.save(os.path.join(run_dir, "mpc_waypoints.npy"), mpc_waypoints)
                with open(os.path.join(run_dir, "mpc_waypoints.json"), "w") as f:
                    json.dump({"waypoints_px": mpc_waypoints.tolist(), "source": "canned"}, f)
                print(f"[mpc] loaded {len(mpc_waypoints)} canned waypoints from {args.canned_waypoints_json}")
            elif args.llm_plan_freq <= 0:
                raise RuntimeError(
                    "guidance_mode=mpc but no waypoint source — set --canned-waypoints-json or --llm-plan-freq>0"
                )

        bar = tqdm.tqdm(range(args.max_timesteps), desc="Running rollout")
        print("Press Ctrl+C to stop early.")

        t_step = 0
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation())
                ext_image = curr_obs[f"{args.external_camera}_image"]
                wrist_image = curr_obs["wrist_image"]
                video_frames.append(ext_image)

                # Optional LLM re-plan (pre-policy so the refined chunk uses the new arrow).
                if (
                    args.guidance_mode == "mpc"
                    and not using_canned_waypoints
                    and args.llm_plan_freq > 0
                    and (mpc_waypoints is None or (t_step > 0 and t_step % args.llm_plan_freq == 0))
                ):
                    _t0 = time.time()
                    new_wp = _llm_plan_waypoints(args, ext_image, instruction)
                    plan_wall = time.time() - _t0
                    if new_wp is not None:
                        mpc_waypoints = new_wp
                        plan_count += 1
                        np.save(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.npy"), mpc_waypoints)
                        with open(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.json"), "w") as f:
                            json.dump({
                                "waypoints_px": mpc_waypoints.tolist(),
                                "t_step": t_step,
                                "planning_wall_seconds": round(plan_wall, 2),
                                "source": "llm_inline",
                            }, f)
                        print(f"[mpc/plan] t={t_step} plan_count={plan_count} wall={plan_wall:.1f}s")
                    else:
                        print(f"[mpc/plan] t={t_step} planner returned None; keeping previous waypoints")

                # 224x224 model inputs (same as main_pi05.py).
                model_input_ext = image_tools.resize_with_pad(ext_image, 224, 224)
                model_input_wrist = image_tools.resize_with_pad(wrist_image, 224, 224)

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

                    # --- MPC overlay refinement (client-side CEM) ------------
                    if args.guidance_mode == "mpc" and mpc_waypoints is not None:
                        _mpc_start = time.time()
                        pred_action_chunk = _refine_chunk_with_mpc(
                            args=args,
                            pred_action_chunk=np.asarray(pred_action_chunk),
                            q0_7=curr_obs["joint_position"],
                            waypoints_px=mpc_waypoints,
                            device=mpc_device,
                        )
                        inference_ms += (time.time() - _mpc_start) * 1000

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper (same as main_pi05.py).
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                action = np.clip(action, -1, 1)

                env.step(action)

                # Per-step log (adds q=[...] like the simulator so the same FK
                # annotator reads real-robot runs identically).
                with open(action_log_path, "a") as alf:
                    alf.write(
                        f"t={t_step:04d} | "
                        f"queried={int(queried_this_step)} | "
                        f"action={np.round(action, 3).tolist()} | "
                        f"gripper={action[-1]:.2f} | "
                        f"chunk_idx={actions_from_chunk_completed}/{args.open_loop_horizon} | "
                        f"inf_ms={inference_ms:.0f} | "
                        f"q={np.round(curr_obs['joint_position'], 5).tolist()}\n"
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
                video_path = os.path.join(run_dir, "rollout.mp4")
                ImageSequenceClip(list(np.stack(video_frames)), fps=10).write_videofile(
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
            f.write(f"policy: pi05_droid (base) + {'mpc_overlay' if args.guidance_mode == 'mpc' else 'baseline'}\n")

        if input("Run another? (y/n): ").strip().lower() != "y":
            break
        env.reset()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
