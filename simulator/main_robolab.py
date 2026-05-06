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
import json
import os
import signal
import sys
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm

# Ensure the client-root (sibling of simulator/) is on sys.path so we can import
# the shared mpc_overlay library without installing it.
_CLIENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CLIENT_ROOT not in sys.path:
    sys.path.insert(0, _CLIENT_ROOT)

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

    # Action space the server emits.
    #   "jointvel" — normalised joint-velocity output (e.g. checkpoint
    #     `pi05_droid` from gs://openpi-assets). Each step is integrated
    #     into a position target via DROID_VEL_LIMITS / 15 Hz. This is the
    #     legacy default and matches RoboLab's `Pi0DroidJointvelClient`.
    #   "jointpos" — joint-position output (e.g. checkpoint
    #     `pi05_droid_jointpos` from gs://openpi-assets-simeval, used in the
    #     RoboLab benchmark). Each step is consumed directly as a 7-DoF
    #     position target. Matches RoboLab's `Pi0DroidJointposClient`.
    action_space: str = "jointvel"

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    fake_policy: bool = False

    # Output
    save_dir: str = "runs_robolab"
    save_frames: bool = False

    # --- MPC-overlay guidance (all optional) -------------------------------
    # "off" = pure pi 0.5, byte-identical to the baseline rollout.
    # "mpc" = after each policy_client.infer() we run CEM locally over the
    #          returned a_vla, using user waypoints as a soft 2D-pixel target.
    guidance_mode: str = "off"

    # Waypoint source (only read when guidance_mode == "mpc"):
    #   if canned_waypoints_json is set, load {"waypoints_px": [[u,v], ...]}
    #   from that file once at the start of the rollout (no LLM).
    #   Otherwise, plan once via GPT + Gemini ER from the first observation.
    canned_waypoints_json: str = ""
    gpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-robotics-er-1.6-preview"

    # MPC weights
    mpc_lam_p: float = 1.0
    mpc_lam_a: float = 1.0
    mpc_lam_c: float = 100.0
    mpc_lam_s: float = 0.01

    # CEM hyperparameters
    mpc_n_samples: int = 200
    mpc_n_iterations: int = 4
    mpc_n_elites: int = 20
    mpc_init_std: float = 0.05

    # MPC compute device. "auto" picks the first CUDA device visible to this
    # process (Isaac Sim's CUDA_VISIBLE_DEVICES selects it), falling back to
    # "cpu" if no CUDA is available. Override with "cpu" or "cuda:N".
    mpc_device: str = "auto"

    # Re-plan the LLM trajectory every N env steps (0 = plan once at t=0).
    # LLM calls take ~10-15 s each, so small values will stall the rollout.
    # Irrelevant when --canned-waypoints-json is set.
    plan_freq: int = 100

    # Multi-step planning (eva_pal_share trajectory_v1 pattern, 2026-05-04).
    # When True: instruction is decomposed into a list of steps; each step
    # gets its OWN trajectory; periodic Gemini step-completion checks advance
    # the state machine to the next step. Fixes failure modes where a single
    # arrow over the whole task can't cover multi-pickup scenes (e.g.
    # `BananasInCrateTask` — the v_arb_lam3 regression).
    enable_multistep: bool = False
    # Run a Gemini "is current step complete?" check every N env steps.
    # Each check is one Gemini call (~5-10s) + up to 3 frames; keep large.
    multistep_check_interval: int = 30

    # When True, paint the LLM-planned trajectory (mpc_waypoints) onto the
    # external camera image *before* sending to the policy server, using the
    # same training-time TraceOverlayConfig as `main_robolab_overlay.py`.
    # Required when serving the trajectory_overlay finetune
    # (`brandonyang/pi05_droid_trajectory_overlay`) — that model was
    # trained to follow a drawn arrow, so the input image must contain one.
    # No-op when --guidance-mode=off or when waypoints aren't available.
    draw_trajectory_overlay: bool = False

    # Sliding-window arrow target (v2). None = legacy full-arrow target. Set to
    # e.g. 0.15 to ask MPC to advance 15% of the arrow's arc length per chunk.
    mpc_arrow_lookahead: float | None = None
    # Gripper-action reward. 0 = disabled; >0 = reward close-near-start and
    # open-near-end action means. Only has effect if CEM doesn't freeze dim 7.
    mpc_gripper_reward_weight: float = 0.0
    mpc_gripper_zone_frac: float = 0.15
    # Post-CEM gripper override: when EE is within gripper_zone_frac of the arrow
    # start the gripper is forced closed; near the end it is forced open. Handles
    # the grasp-stall mode where VLA closes gripper one step and releases the
    # next. Independent of the reward term -- works with freeze_gripper=True.
    mpc_gripper_force_override: bool = False
    mpc_gripper_force_pixel_zone: float = 0.0
    # Progress reward (MPCC-style): rewards the chunk's ending EE arc-length
    # projection being further along the arrow than its starting projection.
    # 0.0 = disabled. Useful for "pure arrow follower" extreme-mode testing.
    mpc_lam_prog: float = 0.0
    # If False, CEM is allowed to sample dim 7 (gripper) — enables learned grip
    # timing via the gripper-reward term. Default True = legacy behaviour
    # (gripper frozen to the VLA's original choice).
    mpc_freeze_gripper: bool = True
    # --- Policy-blending arbitration (Dragan-Srinivasa) -------------------
    # When > 0, enables an EE-to-closer-endpoint distance check each cost
    # call that scales lam_a / lam_prog by alpha in [0, 1] and adds
    # (1-alpha)*mpc_prior_boost_near_waypoint to lam_p. Hands over control
    # to the VLA's semantic priors near grasp/release. 0 = legacy behaviour.
    mpc_arbitration_d_grasp_px: float = 0.0
    mpc_arbitration_tau_px: float = 15.0
    mpc_prior_boost_near_waypoint: float = 0.0
    # Gripper-state-aware arbitration: > 0 enables α_gripper that goes to 0
    # when gripper closed (state > threshold). 0 = disabled.
    mpc_arbitration_gripper_threshold: float = 0.0
    mpc_arbitration_gripper_tau: float = 0.05
    # --- Trust-region projection on CEM samples ---------------------------
    # When > 0, every CEM sample is projected into an L2 ball of this
    # radius around the VLA prior. Hard cap on how far the refined chunk
    # can drift from a_vla, irrespective of how aggressive the arrow
    # weights become. 0 = legacy behaviour.
    mpc_trust_region_radius: float = 0.0


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
# MPC-overlay planning helpers (only used when guidance_mode == "mpc")
# ---------------------------------------------------------------------------

def _load_canned_waypoints(path: str) -> np.ndarray:
    """Load hand-specified waypoints from ``{"waypoints_px": [[u, v], ...]}``.

    Coords must be in the **raw 1280x720 external-cam frame** (no flip in sim).
    """
    with open(path) as f:
        data = json.load(f)
    wp = np.asarray(data["waypoints_px"], dtype=np.float32)
    if wp.ndim != 2 or wp.shape[-1] != 2 or wp.shape[0] < 2:
        raise ValueError(f"canned waypoints must be (K>=2, 2), got shape {wp.shape}")
    return wp


# --- Subprocess bridge to the isolated google-gen-ai-env venv --------------
# google-genai and isaacsim can't coexist in one venv (websockets version pin),
# so LLM trajectory planning runs in ../google-gen-ai-env/.venv via a CLI
# subprocess. Main process only knows about the path + the output JSON schema.
_LLM_PLANNER_DIR = os.path.abspath(os.path.join(_CLIENT_ROOT, "google-gen-ai-env"))
_LLM_PLANNER_PY = os.path.join(_LLM_PLANNER_DIR, ".venv", "bin", "python")
_LLM_PLANNER_SCRIPT = os.path.join(_LLM_PLANNER_DIR, "plan_waypoints.py")


def _plan_waypoints_via_subprocess(
    pil_image: "Image.Image",
    instruction: str,
    gpt_model: str,
    gemini_model: str,
    timeout_seconds: float = 60.0,
    env_file: str = "/home/asethi04/ROBOTICS/.env",
) -> np.ndarray | None:
    """Invoke ``google-gen-ai-env/plan_waypoints.py`` as a subprocess.

    Writes ``pil_image`` to a temp PNG, spawns the planner CLI, reads back the
    waypoints JSON. Returns ``(K, 2)`` float pixel coords in the input image's
    frame, or ``None`` on any failure (subprocess missing, timeout, non-zero
    exit, JSON says ``ok=False``). All failures log to stdout.
    """
    import subprocess
    import tempfile

    if not os.path.exists(_LLM_PLANNER_PY):
        print(f"[mpc/plan] planner venv python not found at {_LLM_PLANNER_PY}; skipping")
        return None
    if not os.path.exists(_LLM_PLANNER_SCRIPT):
        print(f"[mpc/plan] plan_waypoints.py not found at {_LLM_PLANNER_SCRIPT}; skipping")
        return None

    with tempfile.TemporaryDirectory(prefix="mpc_plan_") as tmpd:
        img_path = os.path.join(tmpd, "frame.png")
        out_path = os.path.join(tmpd, "waypoints.json")
        pil_image.save(img_path)

        cmd = [
            _LLM_PLANNER_PY, _LLM_PLANNER_SCRIPT,
            "--image-path", img_path,
            "--instruction", instruction,
            "--output-json", out_path,
            "--gpt-model", gpt_model,
            "--gemini-model", gemini_model,
            "--env-file", env_file,
        ]
        try:
            result = subprocess.run(
                cmd, timeout=timeout_seconds, capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            print(f"[mpc/plan] LLM subprocess timed out after {timeout_seconds}s")
            return None

        if result.returncode != 0 or not os.path.exists(out_path):
            print(f"[mpc/plan] LLM subprocess failed (rc={result.returncode})")
            if result.stderr:
                print(result.stderr[-500:])
            return None

        try:
            with open(out_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[mpc/plan] failed to read subprocess output JSON: {e!r}")
            return None

        if not data.get("ok", False):
            print(f"[mpc/plan] LLM planner returned error: {data.get('error', 'unknown')}")
            return None

        wp = np.asarray(data["waypoints_px"], dtype=np.float32)
        dt = data.get("planning_time_seconds")
        manip = data.get("manipulating_object", "?")
        tgt = data.get("target_related_object", "?")
        print(
            f"[mpc/plan] LLM returned {len(wp)} waypoints "
            f"({manip!r} -> {tgt!r}) in {dt}s"
        )
        return wp


# --- Multi-step planner subprocess (eva_pal_share trajectory_v1.py pattern) ---
# Usage: see google-gen-ai-env/multistep_planner.py. Three modes:
#   - decompose: instruction + frame -> step list
#   - plan_step: frame + step dict -> trajectory waypoints (replanned per step)
#   - check_completion: frames + step text -> is_complete bool
# Adds GPT/Gemini subprocess overhead per call (~10-20s) so use sparingly.
_MULTISTEP_SCRIPT = os.path.join(_LLM_PLANNER_DIR, "multistep_planner.py")


def _multistep_subprocess(mode: str, *, timeout: float = 60.0, **kwargs) -> dict | None:
    """Generic subprocess wrapper for multistep_planner.py.
    Returns the result dict (with `ok` key) or None on subprocess-level failure."""
    import subprocess
    import tempfile

    if not os.path.exists(_LLM_PLANNER_PY):
        print(f"[multistep] planner venv missing at {_LLM_PLANNER_PY}; skipping")
        return None
    if not os.path.exists(_MULTISTEP_SCRIPT):
        print(f"[multistep] multistep_planner.py not found at {_MULTISTEP_SCRIPT}; skipping")
        return None

    with tempfile.TemporaryDirectory(prefix=f"ms_{mode}_") as tmpd:
        out_path = os.path.join(tmpd, "out.json")
        cmd = [_LLM_PLANNER_PY, _MULTISTEP_SCRIPT,
               "--mode", mode, "--output-json", out_path]

        # Mode-specific args.
        if mode == "decompose":
            img = kwargs["pil_image"]
            img_path = os.path.join(tmpd, "frame.png")
            img.save(img_path)
            cmd += ["--image-path", img_path,
                    "--instruction", kwargs["instruction"],
                    "--gpt-model", kwargs.get("gpt_model", "gpt-4o-mini")]
        elif mode == "plan_step":
            img = kwargs["pil_image"]
            img_path = os.path.join(tmpd, "frame.png")
            img.save(img_path)
            cmd += ["--image-path", img_path,
                    "--step-json", json.dumps(kwargs["step"]),
                    "--instruction", kwargs.get("instruction", ""),
                    "--gpt-model", kwargs.get("gpt_model", "gpt-4o-mini"),
                    "--gemini-model", kwargs.get("gemini_model", "gemini-robotics-er-1.6-preview")]
        elif mode == "check_completion":
            paths = []
            for i, im in enumerate(kwargs["pil_images"]):
                p = os.path.join(tmpd, f"f{i:03d}.png")
                im.save(p)
                paths.append(p)
            cmd += ["--image-paths", *paths,
                    "--step-text", kwargs["step_text"],
                    "--gemini-model", kwargs.get("gemini_model", "gemini-robotics-er-1.6-preview")]
        else:
            print(f"[multistep] unknown mode: {mode}")
            return None

        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"[multistep/{mode}] subprocess timed out after {timeout}s")
            return None
        if result.returncode != 0 or not os.path.exists(out_path):
            print(f"[multistep/{mode}] subprocess failed (rc={result.returncode})")
            if result.stderr:
                print(result.stderr[-500:])
            return None
        try:
            with open(out_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[multistep/{mode}] could not parse output: {e!r}")
            return None
        if not data.get("ok", False):
            print(f"[multistep/{mode}] error: {data.get('error', 'unknown')}")
            return None
        return data


def _make_mpc_runtime(args: "Args"):
    """Build the (weights, cem_params) pair once per rollout; reused every step.

    Returns ``(MPCWeights, CEMParams)`` or ``(None, None)`` when mpc mode is off.
    """
    if args.guidance_mode != "mpc":
        return None, None
    from mpc_overlay import CEMParams, MPCWeights
    weights = MPCWeights(
        lam_p=float(args.mpc_lam_p),
        lam_a=float(args.mpc_lam_a),
        lam_c=float(args.mpc_lam_c),
        lam_s=float(args.mpc_lam_s),
        lam_prog=float(args.mpc_lam_prog),
        arbitration_d_grasp_px=float(args.mpc_arbitration_d_grasp_px),
        arbitration_tau_px=float(args.mpc_arbitration_tau_px),
        prior_boost_near_waypoint=float(args.mpc_prior_boost_near_waypoint),
        arbitration_gripper_threshold=float(args.mpc_arbitration_gripper_threshold),
        arbitration_gripper_tau=float(args.mpc_arbitration_gripper_tau),
    )
    cem_params = CEMParams(
        n_samples=int(args.mpc_n_samples),
        n_iterations=int(args.mpc_n_iterations),
        n_elites=int(args.mpc_n_elites),
        init_std=float(args.mpc_init_std),
        freeze_gripper=bool(args.mpc_freeze_gripper),
        trust_region_radius=float(args.mpc_trust_region_radius),
    )
    return weights, cem_params


def _resolve_mpc_device(requested: str) -> "torch.device":
    """Resolve --mpc-device. 'auto' → first visible CUDA, else cpu."""
    import torch
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(requested)


def _refine_chunk_with_mpc(
    env,
    pred_action_chunk: np.ndarray,  # (T, 8) float, as returned by the server
    q0_7: np.ndarray,                # (7,) current joint positions
    waypoints_px: np.ndarray,        # (K, 2) raw-frame pixels
    weights,                         # MPCWeights
    cem_params,                      # CEMParams
    device,                          # torch.device for MPC tensors
    env_id: int = 0,
    arrow_lookahead: float | None = None,
    gripper_reward_weight: float = 0.0,
    gripper_zone_frac: float = 0.15,
    gripper_force_override: bool = False,
    gripper_state_now: float | None = None,
    gripper_force_pixel_zone: float = 0.0,
    action_space: str = "jointvel",
) -> np.ndarray:
    """Run client-side CEM over the server's chunk. Returns (T, 8) numpy.

    For ``action_space == "jointpos"`` (e.g. pi05_droid_jointpos): the server's
    chunk is in joint-position space. We delta-encode it into a "fake velocity"
    chunk where each step's first 7 dims are absolute position deltas in
    radians, and run the MPC with ``joint_vel_scale=ones(7), control_dt=1.0``
    so the existing ``q_traj = q0 + cumsum(a*jvs*dt)`` integration yields the
    same positions back. After MPC, we re-encode the optimized chunk to
    absolute positions (cumulative sum starting from q0). This keeps every
    line of mpc_overlay/ unchanged.
    """
    import torch  # local import: torch only needed in mpc path
    from mpc_overlay import mpc_overlay
    from simulator.mpc_sim_adapter import build_sim_guidance_spec

    if action_space == "jointpos":
        # Convert position chunk -> normalised "velocity" chunk so MPC's
        # standard machinery (which integrates a*jvs*dt) reconstructs the
        # original positions. Using DROID_VEL_LIMITS/15 Hz keeps CEM's
        # init_std=0.05 producing the *same* per-step position drift as
        # jointvel mode (~0.007 rad/step), which prevents CEM from
        # blowing past the VLA prior. Action-box must be disabled (lam_c=0)
        # because the model's first chunk step can be a large absolute jump
        # from current pose (e.g. ~2 rad) which exceeds [-1, 1] norm.
        pos = np.asarray(pred_action_chunk[..., :7], dtype=np.float32)
        q0 = np.asarray(q0_7, dtype=np.float32)
        deltas = np.empty_like(pos)
        deltas[0] = pos[0] - q0
        deltas[1:] = pos[1:] - pos[:-1]
        # Convert deltas to normalised velocities: v_norm = delta / (jvs * dt)
        jvs_arr = DROID_VEL_LIMITS.astype(np.float32)
        dt = 1.0 / float(SIM_CONTROL_FREQUENCY)
        v_norm = deltas / (jvs_arr * dt)[None, :]
        a_vla_np = np.concatenate([v_norm, np.asarray(pred_action_chunk[..., 7:8])], axis=-1)
        spec = build_sim_guidance_spec(
            env, waypoints_px=waypoints_px, q0_7=q0_7, env_id=env_id,
            arrow_lookahead=arrow_lookahead,
            gripper_reward_weight=gripper_reward_weight,
            gripper_zone_frac=gripper_zone_frac,
            gripper_force_override=gripper_force_override,
            gripper_state_now=gripper_state_now,
            gripper_force_pixel_zone=gripper_force_pixel_zone,
        )
    else:
        a_vla_np = np.asarray(pred_action_chunk).copy()
        spec = build_sim_guidance_spec(
            env, waypoints_px=waypoints_px, q0_7=q0_7, env_id=env_id,
            arrow_lookahead=arrow_lookahead,
            gripper_reward_weight=gripper_reward_weight,
            gripper_zone_frac=gripper_zone_frac,
            gripper_force_override=gripper_force_override,
            gripper_state_now=gripper_state_now,
            gripper_force_pixel_zone=gripper_force_pixel_zone,
        )

    a_vla = torch.as_tensor(a_vla_np.copy(), dtype=torch.float32, device=device)
    a_opt = mpc_overlay(a_vla, spec, weights, cem_params)
    a_opt_np = a_opt.detach().cpu().numpy().astype(pred_action_chunk.dtype)

    if action_space == "jointpos":
        # Re-encode normalised velocities -> absolute positions:
        # pos_opt[k] = q0 + sum_{j<=k} v_norm_opt[j] * jvs * dt
        v_norm_opt = a_opt_np[..., :7]
        jvs_arr = DROID_VEL_LIMITS.astype(pred_action_chunk.dtype)
        dt = 1.0 / float(SIM_CONTROL_FREQUENCY)
        deltas_opt = v_norm_opt * (jvs_arr * dt)[None, :]
        pos_opt = np.cumsum(deltas_opt, axis=0) + np.asarray(q0_7, dtype=pred_action_chunk.dtype)[None, :]
        a_opt_np = np.concatenate([pos_opt, a_opt_np[..., 7:8]], axis=-1)

    return a_opt_np


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
        spec_snapshot_saved = False  # guidance_spec_snapshot.json written once per run

        # MPC runtime (weights + CEM params) — built once per rollout.
        mpc_weights, mpc_cem_params = _make_mpc_runtime(args)
        mpc_waypoints = None  # (K, 2) pixels in the raw external-cam frame
        mpc_device = None
        if args.guidance_mode == "mpc":
            mpc_device = _resolve_mpc_device(args.mpc_device)
            print(f"[mpc] device={mpc_device}")

        # Multi-step planner state (eva_pal_share trajectory_v1 pattern).
        # Populated lazily on first planning trigger when --enable-multistep is set.
        ms_steps: list[dict] | None = None     # full decomposed step list
        ms_step_idx: int = 0                   # which step we're currently on
        ms_step_completed: bool = False        # set True once we've advanced past the last step
        ms_completion_buffer: list = []        # PIL images since the last completion check, capped
        ms_last_check_t: int = -10**9          # t_step at which we last ran a completion check
        MS_BUFFER_CAP = 8                      # at most this many recent frames go to Gemini

        # RoboLab examples reset twice before a rollout to let physics settle.
        obs, _ = env.reset()
        obs, _ = env.reset()

        # --- MPC planning: canned waypoints are loaded once up front.
        # LLM planning runs inside the rollout loop (see below) so re-plans use
        # the current frame, not the pre-rollout initial frame.
        using_canned_waypoints = False
        plan_count = 0  # number of LLM plans fired so far (persisted artifacts are indexed by this)
        if args.guidance_mode == "mpc" and args.canned_waypoints_json:
            try:
                mpc_waypoints = _load_canned_waypoints(args.canned_waypoints_json)
                using_canned_waypoints = True
                print(f"[mpc] loaded {len(mpc_waypoints)} canned waypoints from {args.canned_waypoints_json}")
                np.save(os.path.join(run_dir, "mpc_waypoints.npy"), mpc_waypoints)
                with open(os.path.join(run_dir, "mpc_waypoints.json"), "w") as f:
                    json.dump({"waypoints_px": mpc_waypoints.tolist(), "source": "canned"}, f)
            except Exception as e:  # noqa: BLE001
                print(f"[mpc] FAILED to load canned waypoints: {e!r} — falling back to LLM planning")
                mpc_waypoints = None

        auto_success = None  # set from env term/trunc if the episode ends early
        # Use the env's own max_episode_length if it is *larger* than the
        # CLI default — RoboLab tasks can range from 20s to 600s
        # (300–9000 steps at 15 Hz). The CLI default of 600 was set before we
        # discovered the per-task lengths.
        try:
            env_max = int(env.max_episode_length)
        except Exception:
            env_max = args.max_timesteps
        rollout_steps = max(args.max_timesteps, env_max)
        if rollout_steps != args.max_timesteps:
            print(f"[main_robolab] Using env.max_episode_length={env_max} (overrides --max-timesteps={args.max_timesteps})")
        bar = tqdm.tqdm(range(rollout_steps), desc="Running rollout")
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

                # --- Multi-step planning (eva_pal_share trajectory_v1) ------
                # When --enable-multistep is on:
                #   1. On first plan, decompose instruction into ms_steps.
                #   2. Plan trajectory for the CURRENT step only.
                #   3. Periodically (every multistep_check_interval frames),
                #      ask Gemini "is the current step done?" If yes, advance
                #      ms_step_idx and replan for the new step.
                # Falls back to single-arrow plan_waypoints flow when disabled.
                if (
                    args.guidance_mode == "mpc"
                    and not using_canned_waypoints
                    and args.enable_multistep
                ):
                    # 1. First time: decompose instruction.
                    if ms_steps is None and not ms_step_completed:
                        _pil_frame = Image.fromarray(ext_image).convert("RGB")
                        _t0 = time.time()
                        _decomp = _multistep_subprocess(
                            "decompose",
                            pil_image=_pil_frame,
                            instruction=instruction,
                            gpt_model=args.gpt_model,
                        )
                        if _decomp is not None and _decomp.get("steps"):
                            ms_steps = _decomp["steps"]
                            ms_step_idx = 0
                            print(f"[multistep] t={t_step} decomposed into {len(ms_steps)} step(s) "
                                  f"in {time.time()-_t0:.1f}s")
                            for _i, _s in enumerate(ms_steps):
                                print(f"  step {_i}: {_s.get('step', '?')!r}  "
                                      f"manip={_s.get('manipulating_object', '?')}  "
                                      f"target={_s.get('target_location', '?')}")
                            with open(os.path.join(run_dir, "decomposition.json"), "w") as f:
                                json.dump(_decomp, f, indent=2)
                        else:
                            print(f"[multistep] t={t_step} decompose FAILED, falling back to single-step")
                            ms_steps = [{
                                "step": instruction,
                                "manipulating_object": "object",
                                "target_related_object": "",
                                "target_location": "",
                            }]
                            ms_step_idx = 0

                    # 2. Periodic step-completion check.
                    if (ms_steps is not None
                            and not ms_step_completed
                            and ms_step_idx < len(ms_steps)
                            and t_step - ms_last_check_t >= args.multistep_check_interval
                            and len(ms_completion_buffer) > 0):
                        ms_last_check_t = t_step
                        cur_step = ms_steps[ms_step_idx]
                        cur_text = cur_step.get("step", instruction)
                        # send up to MS_BUFFER_CAP recent frames
                        _frames_for_check = ms_completion_buffer[-MS_BUFFER_CAP:]
                        _t0 = time.time()
                        _comp = _multistep_subprocess(
                            "check_completion",
                            pil_images=_frames_for_check,
                            step_text=cur_text,
                            gemini_model=args.gemini_model,
                        )
                        if _comp is not None and _comp.get("is_complete"):
                            print(f"[multistep] t={t_step} step {ms_step_idx} COMPLETE "
                                  f"({cur_text!r}) reasoning={_comp.get('reasoning', '?')[:80]}")
                            ms_step_idx += 1
                            ms_completion_buffer = []
                            mpc_waypoints = None  # force replan for next step
                            if ms_step_idx >= len(ms_steps):
                                ms_step_completed = True
                                print(f"[multistep] all {len(ms_steps)} step(s) complete at t={t_step}")
                        else:
                            print(f"[multistep] t={t_step} step {ms_step_idx} not yet complete "
                                  f"(checked {len(_frames_for_check)} frames in {time.time()-_t0:.1f}s)")

                    # Append current frame to buffer for next check.
                    if ms_steps is not None and not ms_step_completed:
                        ms_completion_buffer.append(Image.fromarray(ext_image).convert("RGB"))
                        if len(ms_completion_buffer) > MS_BUFFER_CAP * 2:
                            ms_completion_buffer = ms_completion_buffer[-MS_BUFFER_CAP:]

                    # 3. Plan trajectory for the CURRENT step. Triggered by:
                    #   - mpc_waypoints is None (just advanced step OR first plan), OR
                    #   - plan_freq elapsed since last plan (re-localize stale step).
                    if (ms_steps is not None
                            and not ms_step_completed
                            and ms_step_idx < len(ms_steps)
                            and (mpc_waypoints is None
                                 or (args.plan_freq > 0 and t_step > 0
                                     and t_step % args.plan_freq == 0))):
                        _pil_frame = Image.fromarray(ext_image).convert("RGB")
                        cur_step = ms_steps[ms_step_idx]
                        _t0 = time.time()
                        _ps = _multistep_subprocess(
                            "plan_step",
                            pil_image=_pil_frame,
                            step=cur_step,
                            instruction=instruction,
                            gpt_model=args.gpt_model,
                            gemini_model=args.gemini_model,
                        )
                        _wall = time.time() - _t0
                        if _ps is not None and _ps.get("waypoints_px"):
                            mpc_waypoints = np.asarray(_ps["waypoints_px"], dtype=np.float32)
                            plan_count += 1
                            np.save(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.npy"), mpc_waypoints)
                            with open(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.json"), "w") as f:
                                json.dump({
                                    "waypoints_px": mpc_waypoints.tolist(),
                                    "t_step": t_step,
                                    "step_index": ms_step_idx,
                                    "step": cur_step,
                                    "planning_wall_seconds": round(_wall, 2),
                                    "source": "multistep_planner",
                                }, f)
                            print(f"[multistep/plan] t={t_step} step={ms_step_idx} "
                                  f"plan_count={plan_count} wall={_wall:.1f}s "
                                  f"({len(mpc_waypoints)} waypoints)")
                        else:
                            print(f"[multistep/plan] t={t_step} step={ms_step_idx} FAILED")

                # --- Legacy single-arrow LLM re-planning (multistep disabled) -
                elif (
                    args.guidance_mode == "mpc"
                    and not using_canned_waypoints
                    and (
                        mpc_waypoints is None
                        or (args.plan_freq > 0 and t_step > 0 and t_step % args.plan_freq == 0)
                    )
                ):
                    _pil_frame = Image.fromarray(ext_image).convert("RGB")
                    _t0 = time.time()
                    _new_wp = _plan_waypoints_via_subprocess(
                        _pil_frame,
                        instruction=instruction,
                        gpt_model=args.gpt_model,
                        gemini_model=args.gemini_model,
                    )
                    _plan_wall = time.time() - _t0
                    if _new_wp is not None:
                        mpc_waypoints = _new_wp
                        plan_count += 1
                        np.save(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.npy"), mpc_waypoints)
                        with open(os.path.join(run_dir, f"mpc_waypoints_{plan_count:03d}.json"), "w") as f:
                            json.dump({
                                "waypoints_px": mpc_waypoints.tolist(),
                                "t_step": t_step,
                                "planning_wall_seconds": round(_plan_wall, 2),
                                "source": "llm_subprocess",
                            }, f)
                        print(f"[mpc/plan] t={t_step} plan_count={plan_count} wall={_plan_wall:.1f}s")
                    else:
                        print(f"[mpc/plan] t={t_step} planner failed; keeping previous waypoints (have={mpc_waypoints is not None})")

                # Optionally draw the LLM-planned trajectory onto the
                # external image before sending to the policy server. The
                # `pi05_droid_trajectory_overlay` finetune was trained to
                # follow a drawn arrow, so it needs to see one. When
                # `--draw-trajectory-overlay` is OFF (default), the raw
                # camera image is used.
                ext_for_policy = ext_image
                if args.draw_trajectory_overlay and mpc_waypoints is not None and len(mpc_waypoints) >= 2:
                    try:
                        from traj_vis_utils import TraceOverlayConfig, add_trace_overlay
                        pts = [tuple(map(float, p)) for p in mpc_waypoints]
                        result = add_trace_overlay(
                            ext_image,
                            pts,
                            current_index=0,
                            config=TraceOverlayConfig(),
                            num_interpolated=100,
                        )
                        ext_for_policy = np.array(result.convert("RGB"))
                    except Exception as _e:
                        if t_step == 0:
                            print(f"[overlay-draw] failed: {_e!r} — falling back to raw frame")

                # Compute 224x224 model inputs up front so we can dump them to
                # disk every step even if we don't query the policy this step.
                # These are the EXACT tensors sent to the policy server.
                model_input_ext = image_tools.resize_with_pad(ext_for_policy, 224, 224)
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

                    # --- MPC overlay refinement (client-side CEM) --------
                    # Runs every time the server returns a fresh chunk, using the
                    # pre-planned waypoints. VLA stays frozen; we only touch the
                    # chunk. If guidance is off or waypoints are missing this
                    # block is a no-op.
                    if args.guidance_mode == "mpc" and mpc_waypoints is not None:
                        _mpc_start = time.time()
                        if not spec_snapshot_saved:
                            try:
                                from simulator.mpc_sim_adapter import (
                                    compute_sim_intrinsics, get_sim_extrinsic_6vec,
                                    DROID_VEL_LIMITS, DROID_CONTROL_HZ,
                                )
                                K_snap = compute_sim_intrinsics().tolist()
                                ext_snap = get_sim_extrinsic_6vec(env).tolist()
                                jvs_snap = DROID_VEL_LIMITS.tolist()
                                with open(os.path.join(run_dir, "guidance_spec_snapshot.json"), "w") as f:
                                    json.dump({
                                        "K_intrinsics": K_snap,
                                        "extrinsic_cam_in_base": ext_snap,
                                        "joint_vel_scale": jvs_snap,
                                        "image_hw": [720, 1280],
                                        "control_dt": 1.0 / DROID_CONTROL_HZ,
                                        "image_flipped_180": False,
                                        "ee_offset_from_flange": 0.1034,
                                    }, f, indent=2)
                                spec_snapshot_saved = True
                            except Exception as _e:
                                print(f"[mpc] spec snapshot failed: {_e!r}")
                        pred_action_chunk = _refine_chunk_with_mpc(
                            env=env,
                            pred_action_chunk=np.asarray(pred_action_chunk),
                            q0_7=curr_obs["joint_position"],
                            waypoints_px=mpc_waypoints,
                            weights=mpc_weights,
                            cem_params=mpc_cem_params,
                            device=mpc_device,
                            arrow_lookahead=args.mpc_arrow_lookahead,
                            gripper_reward_weight=args.mpc_gripper_reward_weight,
                            gripper_zone_frac=args.mpc_gripper_zone_frac,
                            gripper_force_override=args.mpc_gripper_force_override,
                            gripper_state_now=float(np.asarray(curr_obs["gripper_position"]).flatten()[0]),
                            gripper_force_pixel_zone=args.mpc_gripper_force_pixel_zone,
                            action_space=args.action_space,
                        )
                        inference_ms += (time.time() - _mpc_start) * 1000

                    # Re-anchor velocity integrator to the CURRENTLY measured
                    # joint position; bounds open-loop drift to one chunk.
                    q_target = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if args.action_space == "jointpos":
                    # Server emits joint *positions* directly (e.g.
                    # pi05_droid_jointpos checkpoint). Use as-is.
                    q_target = np.asarray(action[:7], dtype=np.float32)
                else:
                    # Velocity → position integration (sim env expects joint-
                    # position targets; the server emits joint velocities).
                    q_target = _integrate_velocity(action[:7], q_target)

                # Binarize gripper (same threshold as main_pi05.py and
                # both RoboLab Pi0Droid clients).
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
                        f"inf_ms={inference_ms:.0f} | "
                        f"q={np.round(curr_obs['joint_position'], 5).tolist()}\n"
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

        # If the Python for-loop ran to completion without `term` or `trunc`
        # firing, the task is still unfinished at max_timesteps. Count that as
        # an explicit failure rather than leaving auto_success=None, so result.txt
        # distinguishes rollouts from tasks whose env doesn't truncate on its own.
        if auto_success is None:
            auto_success = False
            print(f"[main_robolab] Reached max_timesteps ({args.max_timesteps}) without env termination; auto_success=False.")

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
        # Only prompt for human label when an interactive TTY is attached;
        # otherwise trust auto_success to avoid contaminating result.txt with
        # whatever happened to be piped on stdin (e.g. `printf '\ny\nn\n' | ...`
        # inside a sweep harness will answer 'y' for every rollout).
        if sys.stdin.isatty():
            try:
                success = input("Did the rollout succeed? (y/n): ").strip().lower()
            except EOFError:
                success = "auto" if auto_success is None else ("y" if auto_success else "n")
        else:
            success = "auto" if auto_success is None else ("y" if auto_success else "n")
            print(f"[main_robolab] non-interactive; success label = {success} (from auto_success)")

        with open(os.path.join(run_dir, "result.txt"), "w") as f:
            f.write(f"success: {success}\n")
            f.write(f"auto_success: {auto_success}\n")
            f.write(f"success_source: {'human_prompt' if sys.stdin.isatty() else 'auto_detector'}\n")
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
    parser.add_argument(
        "--action-space", "--action_space", type=str, default="jointvel",
        choices=["jointvel", "jointpos"],
        help="jointvel (legacy): integrate normalized joint velocities * vel_limits / 15Hz. "
             "jointpos: server emits joint-position targets directly (matches RoboLab's "
             "Pi0DroidJointposClient + pi05_droid_jointpos checkpoint).",
    )

    # --- MPC-overlay guidance (off by default → baseline behavior) --------
    parser.add_argument(
        "--guidance-mode", "--guidance_mode", type=str, default="off", choices=["off", "mpc"],
        help="off: baseline pi 0.5 (byte-identical to prior runs). mpc: client-side CEM refinement of the server's chunk.",
    )
    parser.add_argument(
        "--canned-waypoints-json", "--canned_waypoints_json", type=str, default="",
        help='JSON file containing {"waypoints_px": [[u, v], ...]} in the raw 1280x720 external-cam frame. If unset, the LLM planner runs.',
    )
    parser.add_argument("--gpt-model", "--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--gemini-model", "--gemini_model", type=str, default="gemini-robotics-er-1.6-preview")
    parser.add_argument("--mpc-lam-p", "--mpc_lam_p", type=float, default=1.0)
    parser.add_argument("--mpc-lam-a", "--mpc_lam_a", type=float, default=1.0)
    parser.add_argument("--mpc-lam-c", "--mpc_lam_c", type=float, default=100.0)
    parser.add_argument("--mpc-lam-s", "--mpc_lam_s", type=float, default=0.01)
    parser.add_argument("--mpc-n-samples", "--mpc_n_samples", type=int, default=200)
    parser.add_argument("--mpc-n-iterations", "--mpc_n_iterations", type=int, default=4)
    parser.add_argument("--mpc-n-elites", "--mpc_n_elites", type=int, default=20)
    parser.add_argument("--mpc-init-std", "--mpc_init_std", type=float, default=0.05)
    parser.add_argument(
        "--mpc-device", "--mpc_device", type=str, default="auto",
        help='torch device for MPC tensors. "auto" = first CUDA if available else cpu; or "cpu" / "cuda:N".',
    )
    parser.add_argument(
        "--plan-freq", "--plan_freq", type=int, default=100,
        help="Re-plan LLM waypoints every N env steps (0 = once at t=0). Ignored if --canned-waypoints-json is set.",
    )
    parser.add_argument(
        "--mpc-arrow-lookahead", "--mpc_arrow_lookahead", type=float, default=None,
        help="Sliding-window lookahead: target is arrow[s0, s0+lookahead*total] where s0 is current EE projection. Typical 0.10-0.25. Leave unset for legacy full-arrow target.",
    )
    parser.add_argument(
        "--mpc-gripper-reward-weight", "--mpc_gripper_reward_weight", type=float, default=0.0,
        help="Reward gripper closure near arrow start and release near arrow end. 0 = disabled.",
    )
    parser.add_argument(
        "--mpc-gripper-zone-frac", "--mpc_gripper_zone_frac", type=float, default=0.15,
        help="Arc-length fraction at each end of the arrow defining the gripper-reward zones (default 0.15 = first/last 15%).",
    )
    parser.add_argument(
        "--mpc-gripper-force-pixel-zone", "--mpc_gripper_force_pixel_zone", type=float, default=0.0,
        help="When > 0 AND --mpc-gripper-force-override is set, force gripper close when EE pixel is within this many pixels of arrow START, open when within zone of END. Replaces legacy arc-length-based override.",
    )
    parser.add_argument(
        "--mpc-gripper-force-override", "--mpc_gripper_force_override", action="store_true", default=False,
        help="Post-CEM hard override: force gripper closed when EE is near arrow start, open near end. Addresses grasp-stall failure mode.",
    )
    parser.add_argument(
        "--mpc-lam-prog", "--mpc_lam_prog", type=float, default=0.0,
        help="Progress-reward weight: rewards the chunk's final EE projection being further along the arrow than its starting projection. 0 = disabled.",
    )
    parser.add_argument(
        "--mpc-unfreeze-gripper", "--mpc_unfreeze_gripper", action="store_true", default=False,
        help="Let CEM sample the gripper dimension (default: frozen to VLA value). Paired with --mpc-gripper-reward-weight, the CEM can learn grip timing.",
    )
    parser.add_argument(
        "--mpc-arbitration-d-grasp-px", "--mpc_arbitration_d_grasp_px", type=float, default=0.0,
        help="Policy-blending: when EE pixel is within this distance of the closer arrow endpoint, scale lam_a/lam_prog by alpha=sigmoid((d - this)/tau). 0 = off.",
    )
    parser.add_argument(
        "--mpc-arbitration-tau-px", "--mpc_arbitration_tau_px", type=float, default=15.0,
        help="Sharpness of the alpha-arbitration sigmoid. Smaller = sharper transition near d_grasp.",
    )
    parser.add_argument(
        "--mpc-prior-boost-near-waypoint", "--mpc_prior_boost_near_waypoint", type=float, default=0.0,
        help="Additive boost to lam_p when alpha→0 (EE near a critical waypoint). 0 = no boost.",
    )
    parser.add_argument(
        "--mpc-trust-region-radius", "--mpc_trust_region_radius", type=float, default=0.0,
        help="Hard L2 cap on how far a CEM sample may deviate from a_vla. 0 = off.",
    )
    parser.add_argument(
        "--mpc-arbitration-gripper-threshold", "--mpc_arbitration_gripper_threshold",
        type=float, default=0.0,
        help="When > 0, multiplies arbitration alpha by sigmoid((threshold - gripper_state_now)/gripper_tau). When gripper closed, MPC arrow-pull is gated off (let VLA do placement). 0 = off.",
    )
    parser.add_argument(
        "--mpc-arbitration-gripper-tau", "--mpc_arbitration_gripper_tau",
        type=float, default=0.05,
        help="Sigmoid sharpness for the gripper-state arbitration. Smaller = sharper transition.",
    )
    # Multi-step planning (eva_pal_share trajectory_v1 pattern).
    parser.add_argument(
        "--enable-multistep", "--enable_multistep", action="store_true",
        help="Decompose instruction into steps; replan trajectory per step using Gemini step-completion checks.",
    )
    parser.add_argument(
        "--multistep-check-interval", "--multistep_check_interval", type=int, default=30,
        help="Run a step-completion check every N env steps (default 30).",
    )
    parser.add_argument(
        "--draw-trajectory-overlay", "--draw_trajectory_overlay", action="store_true",
        help="Paint LLM trajectory onto external image before sending to policy "
             "(needed when serving the trajectory_overlay finetune).",
    )
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
            action_space=ns.action_space,
            guidance_mode=ns.guidance_mode,
            canned_waypoints_json=ns.canned_waypoints_json,
            gpt_model=ns.gpt_model,
            gemini_model=ns.gemini_model,
            mpc_lam_p=ns.mpc_lam_p,
            mpc_lam_a=ns.mpc_lam_a,
            mpc_lam_c=ns.mpc_lam_c,
            mpc_lam_s=ns.mpc_lam_s,
            mpc_n_samples=ns.mpc_n_samples,
            mpc_n_iterations=ns.mpc_n_iterations,
            mpc_n_elites=ns.mpc_n_elites,
            mpc_init_std=ns.mpc_init_std,
            mpc_device=ns.mpc_device,
            plan_freq=ns.plan_freq,
            enable_multistep=ns.enable_multistep,
            multistep_check_interval=ns.multistep_check_interval,
            draw_trajectory_overlay=ns.draw_trajectory_overlay,
            mpc_arrow_lookahead=ns.mpc_arrow_lookahead,
            mpc_gripper_reward_weight=ns.mpc_gripper_reward_weight,
            mpc_gripper_zone_frac=ns.mpc_gripper_zone_frac,
            mpc_gripper_force_override=ns.mpc_gripper_force_override,
            mpc_gripper_force_pixel_zone=ns.mpc_gripper_force_pixel_zone,
            mpc_lam_prog=ns.mpc_lam_prog,
            mpc_freeze_gripper=(not ns.mpc_unfreeze_gripper),
            mpc_arbitration_d_grasp_px=ns.mpc_arbitration_d_grasp_px,
            mpc_arbitration_tau_px=ns.mpc_arbitration_tau_px,
            mpc_prior_boost_near_waypoint=ns.mpc_prior_boost_near_waypoint,
            mpc_trust_region_radius=ns.mpc_trust_region_radius,
            mpc_arbitration_gripper_threshold=ns.mpc_arbitration_gripper_threshold,
            mpc_arbitration_gripper_tau=ns.mpc_arbitration_gripper_tau,
        ))
    finally:
        _SIM_APP.close()
