"""RoboLab inference client with TRAJECTORY-OVERLAY guidance.

This is the original "draw a trajectory on the image and let a finetuned
VLA follow it" pipeline (real-robot main.py), ported to the RoboLab sim.

Pipeline per rollout:
  1. Boot Isaac Sim + RoboLab task.
  2. Decompose the task instruction into manipulation steps via GPT.
  3. Every plan_freq env steps (and at t=0):
       a. Detect manipulating + target objects via Gemini-ER (pixel coords).
       b. Generate a trajectory of pixel waypoints via GPT.
  4. Each step, overlay the trajectory on the exterior image (red→pink line
     + yellow current-position dot, exact training-time TraceOverlayConfig).
  5. Resize annotated 1280×720 frame to 224×224 and send to the policy
     server. The server is a pi0.5 finetune trained on overlay-annotated
     DROID frames (e.g. brandonyang/pi05_chris_traces_from_droid20k), so
     the model "follows the line" visually.

This is exactly the real-robot main.py flow — no MPC, no CEM, no
client-side guidance. The model does all the work; the LLMs just provide
high-quality visual hints.

Usage:
    python -u main_robolab_overlay.py \
        --task BananaInBowlTask \
        --remote-host 127.0.0.1 --remote-port 8002 \
        --max-timesteps 300 --plan-freq 150 \
        --save-dir ~/runs_overlay --save-frames --headless
"""

import argparse
import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import sys
import time
import traceback

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm

# Make client root importable for traj_vis_utils + trajectory_predictor.
_CLIENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CLIENT_ROOT not in sys.path:
    sys.path.insert(0, _CLIENT_ROOT)


# ---------------------------------------------------------------------------
# Args dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    task: str
    instruction_type: str = "default"
    task_dirs: tuple = ("benchmark",)
    max_timesteps: int = 300
    open_loop_horizon: int = 8
    num_envs: int = 1
    remote_host: str = "127.0.0.1"
    remote_port: int = 8002
    save_dir: str = "runs_overlay"
    save_frames: bool = False
    headless: bool = True
    plan_freq: int = 150
    max_plan_count: int = 20
    gpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    # Multi-step replanning (eva_pal_share trajectory_v1 pattern).
    # Every `step_check_interval` env steps, ask Gemini whether the current
    # decomposed step is complete; on `is_complete=True`, advance step_idx
    # and force a replan of the arrow against the NEXT step's objects.
    # Fixes failure on multi-pickup tasks like `appleandyogurtinbowl` where
    # the arrow stayed pointed at object 1 even after it was placed.
    # 0 disables multistep advancement (legacy behaviour).
    step_check_interval: int = 30
    step_check_max_images: int = 3


# ---------------------------------------------------------------------------
# Ctrl-C protection while a server call is in flight (matches main_robolab.py)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = []

    def _handler(signum, frame):
        interrupted.append((signum, frame))

    old = signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)
        if interrupted:
            signum, frame = interrupted[-1]
            old(signum, frame) if callable(old) else None


# ---------------------------------------------------------------------------
# RoboLab observation → flat dict (matches main_robolab.py byte-for-byte)
# ---------------------------------------------------------------------------
def _extract_observation(obs_dict: dict, env_id: int = 0) -> dict:
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
# Joint-velocity → joint-position integrator (matches RoboLab's Pi0DroidJointvelClient)
# ---------------------------------------------------------------------------
DROID_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
DROID_CONTROL_HZ = 15.0


def _integrate_velocity(action: np.ndarray, q_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """action ∈ R^8 with [qd0..qd6 (normalized to ±1), gripper], q_target ∈ R^7 current target.
    Returns new (q_target_7, gripper_scalar_or_array)."""
    qd_norm = np.asarray(action[:7], dtype=np.float32)
    qd = qd_norm * DROID_VEL_LIMITS
    new_q = q_target + qd / DROID_CONTROL_HZ
    return new_q.astype(np.float32), float(action[7])


# ---------------------------------------------------------------------------
# LLM planning state (mirrors main.py's PlanningState exactly)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PlanningState:
    steps: list[dict] = dataclasses.field(default_factory=list)
    step_idx: int = 0
    pred_traj: dict | None = None
    current_end_point: list[float] | None = None
    img_history: list = dataclasses.field(default_factory=list)
    plan_count: int = 0
    steps_since_last_plan: int = 0
    full_task: str = ""
    _last_object_locations: dict | None = None

    def reset_step(self):
        self.pred_traj = None
        self.current_end_point = None
        self.img_history = []
        self.steps_since_last_plan = 0

    @property
    def current_step(self):
        if self.step_idx < len(self.steps):
            return self.steps[self.step_idx]
        return None

    @property
    def all_steps_done(self) -> bool:
        return self.step_idx >= len(self.steps)


def _decompose_instruction(instruction: str, ext_image_rgb: np.ndarray, args: Args) -> list[dict]:
    """Run GPT-4 instruction decomposition. Returns list of step dicts:
       [{step, manipulating_object, target_related_object, target_location}, ...]"""
    from trajectory_predictor import query_target_objects, encode_pil_image, resize_for_api
    pil = resize_for_api(Image.fromarray(ext_image_rgb).convert("RGB"))
    img_b64 = encode_pil_image(pil)
    out = query_target_objects(instruction, gpt_model=args.gpt_model, img_encoded=img_b64)
    return out.get("steps", [])


def _predict_trajectory_for_step(planning: PlanningState, ext_image_rgb: np.ndarray,
                                  args: Args) -> dict | None:
    """Borrowed from main.py._predict_trajectory but inlined and trimmed for sim use."""
    from trajectory_predictor import (
        query_target_location, query_trajectory, encode_pil_image, resize_for_api,
        _fuzzy_find,
    )

    step = planning.current_step
    if step is None:
        return None
    manipulating_object = step.get("manipulating_object", "")
    target_related_object = step.get("target_related_object", "")
    target_location = step.get("target_location", "")

    detect_queries = list({q for q in (manipulating_object, target_related_object) if q.strip()})
    if not detect_queries:
        return None

    pil_img = Image.fromarray(ext_image_rgb).convert("RGB")
    original_size = pil_img.size
    api_img = resize_for_api(pil_img, max_size=1024)
    img_b64 = encode_pil_image(api_img)

    # Run Gemini for object detection. query_target_location returns dict
    # {object_name: (x_api, y_api)}.
    locs_api = query_target_location(api_img, detect_queries, model_name=args.gemini_model) or {}
    planning._last_object_locations = locs_api

    manip_pt_api = _fuzzy_find(locs_api, manipulating_object)
    target_pt_api = _fuzzy_find(locs_api, target_related_object) if target_related_object else None
    if manip_pt_api is None:
        print(f"[plan] Gemini missed manipulating object {manipulating_object!r}; abort plan.")
        return None

    # Rescale API-coords back to original image coords
    api_w, api_h = api_img.size
    orig_w, orig_h = original_size
    sx = orig_w / api_w
    sy = orig_h / api_h
    manip_pt = (manip_pt_api[0] * sx, manip_pt_api[1] * sy)
    if target_pt_api is not None:
        target_pt_orig = (target_pt_api[0] * sx, target_pt_api[1] * sy)
    else:
        target_pt_orig = None

    traj = query_trajectory(
        img=api_img,
        img_encoded=img_b64,
        task=step.get("step", planning.full_task),
        manipulating_object=manipulating_object,
        manipulating_object_point=manip_pt_api,
        target_related_object=target_related_object or "",
        target_related_object_point=target_pt_api or (api_w / 2, api_h / 2),
        target_location=target_location or "",
        gpt_model=args.gpt_model,
        target_location_point=None,
        full_task=planning.full_task,
    )

    # Rescale all points back to original frame coords
    if traj.get("trajectory"):
        traj["trajectory"] = [[p[0] * sx, p[1] * sy] for p in traj["trajectory"]]
    if traj.get("start_point"):
        sp = traj["start_point"]
        traj["start_point"] = [sp[0] * sx, sp[1] * sy]
    if traj.get("end_point"):
        ep = traj["end_point"]
        traj["end_point"] = [ep[0] * sx, ep[1] * sy]
        planning.current_end_point = traj["end_point"]
    return traj


def _draw_trajectory_on_image(image_rgb: np.ndarray, trajectory_points: list[list[float]],
                               current_index: int = 0):
    """Draw trajectory overlay matching the training-time TraceOverlayConfig."""
    from traj_vis_utils import TraceOverlayConfig, add_trace_overlay
    if len(trajectory_points) < 2:
        return image_rgb
    pts = [tuple(p) for p in trajectory_points]
    idx = min(current_index, len(pts) - 1)
    config = TraceOverlayConfig()
    result = add_trace_overlay(image_rgb, pts, current_index=idx, config=config,
                               num_interpolated=100)
    return np.array(result.convert("RGB"))


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------
def main(args: Args, app):
    """app: the Isaac Sim AppLauncher.app handle (so we can close it on exit)."""
    import torch
    from robolab.core.environments.runtime import create_env
    from robolab.core.logging.recorder_manager import patch_recorder_manager
    from robolab.registrations.droid_jointpos.auto_env_registrations import (
        auto_register_droid_envs,
    )

    patch_recorder_manager()
    auto_register_droid_envs(task_dirs=list(args.task_dirs), task=args.task)
    env, env_cfg = create_env(args.task, num_envs=args.num_envs, use_fabric=True)

    obs, _ = env.reset()
    obs, _ = env.reset()
    print(f"[overlay] env created: {args.task}; default instruction: {env_cfg.instruction!r}")

    # Connect to policy server
    try:
        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    except Exception as e:
        print(f"[overlay] policy server unreachable at {args.remote_host}:{args.remote_port} — {e!r}")
        return 2

    # Run per-rollout
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    run_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    frames_dir = os.path.join(run_dir, "frames")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    instruction = env_cfg.instruction
    with open(os.path.join(run_dir, "instruction.txt"), "w") as f:
        f.write(instruction)

    actions_log_path = os.path.join(run_dir, "actions.log")
    actions_log = open(actions_log_path, "w")

    planning = PlanningState()
    planning.full_task = instruction

    # --- Step 1: decompose instruction ---
    print(f"\n[overlay] decomposing: {instruction!r}")
    try:
        first_obs = _extract_observation(obs)
        steps = _decompose_instruction(instruction, first_obs["external_image"], args)
    except Exception as e:
        print(f"[overlay] decomposition FAILED: {e!r}")
        traceback.print_exc()
        steps = []

    if not steps:
        # Fallback: build a single-step plan from the instruction itself
        print("[overlay] no steps decomposed; using single-step fallback")
        steps = [{
            "step": instruction,
            "manipulating_object": "object",
            "target_related_object": "",
            "target_location": "",
        }]

    # Plan-once mode: collapse multi-step into one (simpler in sim)
    if args.max_plan_count == 1 and len(steps) > 1:
        merged = {
            "step": instruction,
            "manipulating_object": steps[0].get("manipulating_object", ""),
            "target_related_object": (steps[-1].get("target_related_object")
                                       or steps[0].get("target_related_object", "")),
            "target_location": (steps[-1].get("target_location")
                                  or steps[0].get("target_location", "")),
        }
        steps = [merged]

    planning.steps = steps
    print(f"[overlay] decomposed into {len(steps)} step(s):")
    for i, s in enumerate(steps):
        print(f"  {i+1}. {s.get('step', '?')}  "
              f"(manip={s.get('manipulating_object', '?')}, "
              f"target={s.get('target_location', s.get('target_related_object', '?'))})")

    with open(os.path.join(run_dir, "decomposition.json"), "w") as f:
        json.dump({"instruction": instruction, "steps": steps}, f, indent=2)

    # Sticky integrator anchor
    obs_extracted = _extract_observation(obs)
    q_target = obs_extracted["joint_position"].copy()

    # --- Step 2: rollout ---
    pred_action_chunk = None
    actions_from_chunk_completed = 0
    # Use the env's native max_episode_length when it exceeds the CLI flag.
    # RoboLab tasks span 20s..600s = 300..9000 steps at 15 Hz; the CLI default
    # of 300 was set before we discovered per-task lengths.
    try:
        env_max = int(env.max_episode_length)
    except Exception:
        env_max = args.max_timesteps
    rollout_steps = max(args.max_timesteps, env_max)
    if rollout_steps != args.max_timesteps:
        print(f"[overlay] using env.max_episode_length={env_max} (overrides --max-timesteps={args.max_timesteps})")
    bar = tqdm.tqdm(range(rollout_steps), desc="Overlay rollout")
    print("[overlay] starting rollout. Press Ctrl-C to stop early.")

    auto_success = False
    last_step_check_t = -10**9  # track when we last ran a completion check
    try:
        for t_step in bar:
            # `obs` is updated by env.step() at the end of the previous iteration
            # (or by env.reset() before this loop). RoboLab env has no get_observation method.
            curr_obs = _extract_observation(obs)
            ext_image = curr_obs["external_image"]
            wrist_image = curr_obs["wrist_image"]

            # Accumulate frames so the step-completion check has temporal context.
            # Bounded buffer: cap at 6× max_images so we don't grow forever.
            if planning.current_step is not None:
                planning.img_history.append(Image.fromarray(ext_image).convert("RGB"))
                _max_buf = max(args.step_check_max_images * 6, 24)
                if len(planning.img_history) > _max_buf:
                    planning.img_history = planning.img_history[-_max_buf:]

            # --- Multi-step completion check (eva_pal_share trajectory_v1) ---
            # Periodically ask Gemini "is the current step done?"; if yes,
            # advance step_idx and force a replan against the NEXT step's
            # objects. Only run when we've had enough frames since the last
            # check AND the current step's plan has been in effect a while.
            step_advanced_this_iter = False
            if (
                args.step_check_interval > 0
                and not planning.all_steps_done
                and planning.current_step is not None
                and (t_step - last_step_check_t) >= args.step_check_interval
                and len(planning.img_history) >= args.step_check_max_images
                and planning.pred_traj is not None  # only check after first plan
            ):
                last_step_check_t = t_step
                from trajectory_predictor import query_step_completion
                cur = planning.current_step
                cur_text = cur.get("step", planning.full_task)
                _frames = planning.img_history[-args.step_check_max_images:]
                try:
                    t0 = time.time()
                    _comp = query_step_completion(_frames, step=cur_text,
                                                   model_name=args.gemini_model,
                                                   max_images=args.step_check_max_images)
                    _wall = time.time() - t0
                    print(f"[step-check] t={t_step} step_idx={planning.step_idx} "
                          f"is_complete={_comp.get('is_complete')} wall={_wall:.1f}s "
                          f"reasoning={(_comp.get('reasoning') or '')[:80]}")
                    if _comp.get("is_complete"):
                        planning.step_idx += 1
                        planning.reset_step()  # clears pred_traj so we replan immediately
                        step_advanced_this_iter = True
                        if planning.all_steps_done:
                            print(f"[step-check] all {len(planning.steps)} steps complete at t={t_step}")
                except Exception as e:
                    print(f"[step-check] failed: {e!r}")

            # --- Planning gate ---
            # Plan when:
            #   - this is t=0, OR
            #   - we just advanced to a new step (pred_traj is None), OR
            #   - plan_freq elapsed since the last plan
            should_plan = (
                planning.plan_count < args.max_plan_count
                and not planning.all_steps_done
                and (
                    t_step == 0
                    or step_advanced_this_iter
                    or planning.pred_traj is None
                    or (args.plan_freq > 0 and t_step % args.plan_freq == 0)
                )
            )
            if should_plan:
                t0 = time.time()
                try:
                    traj = _predict_trajectory_for_step(planning, ext_image, args)
                except Exception as e:
                    print(f"[overlay] plan exception: {e!r}")
                    traceback.print_exc()
                    traj = None
                wall = time.time() - t0
                if traj is not None:
                    planning.pred_traj = traj
                    planning.plan_count += 1
                    planning.steps_since_last_plan = 0
                    print(f"[plan] t={t_step} step_idx={planning.step_idx} "
                          f"count={planning.plan_count} wall={wall:.1f}s "
                          f"waypoints={len(traj.get('trajectory', []))} "
                          f"step={(planning.current_step or {}).get('step', '?')!r}")
                    plan_path = os.path.join(run_dir, f"plan_{planning.plan_count:03d}.json")
                    with open(plan_path, "w") as f:
                        json.dump({
                            "t_step": t_step,
                            "step_idx": planning.step_idx,
                            "step": planning.current_step,
                            "wall": wall,
                            "trajectory": traj,
                        }, f, indent=2)
                else:
                    print(f"[plan] t={t_step} step_idx={planning.step_idx} FAILED (took {wall:.1f}s); will retry next plan_freq")
            else:
                planning.steps_since_last_plan += 1

            # --- 224x224 model inputs ---
            # CRITICAL: draw trajectory on the *resized* image to match
            # training-time pixel scale. DROID training images are 320x180;
            # 3px line + 5px dot are clearly visible after pad+resize to 224.
            # If we draw on 1280x720 first then resize, the line shrinks 4x
            # and effectively disappears at 224 — that was the bug.
            model_input_ext_clean = image_tools.resize_with_pad(ext_image, 224, 224)
            model_input_wrist = image_tools.resize_with_pad(wrist_image, 224, 224)

            annotated_ext = ext_image  # kept for save_frames at full res
            model_input_ext = model_input_ext_clean
            if planning.pred_traj is not None:
                pts = planning.pred_traj.get("trajectory", [])
                if len(pts) >= 2:
                    # Scale trajectory points from camera resolution (H, W) into
                    # the resize_with_pad letterbox on a 224x224 canvas.
                    H, W = ext_image.shape[:2]
                    s = min(224 / W, 224 / H)
                    pad_x = (224 - W * s) / 2
                    pad_y = (224 - H * s) / 2
                    pts_224 = [[p[0] * s + pad_x, p[1] * s + pad_y] for p in pts]
                    model_input_ext = _draw_trajectory_on_image(
                        model_input_ext_clean, pts_224, current_index=0)
                    # Also draw on the full-res frame for visualization
                    annotated_ext = _draw_trajectory_on_image(ext_image, pts, current_index=0)

            if args.save_frames:
                Image.fromarray(ext_image).save(os.path.join(frames_dir, f"{t_step:04d}_ext.jpg"), quality=85)
                Image.fromarray(annotated_ext).save(os.path.join(frames_dir, f"{t_step:04d}_ext_annotated.jpg"), quality=85)
                Image.fromarray(model_input_ext).save(os.path.join(frames_dir, f"{t_step:04d}_ext_224.jpg"), quality=90)
                Image.fromarray(wrist_image).save(os.path.join(frames_dir, f"{t_step:04d}_wrist.jpg"), quality=85)

            # --- Inference ---
            queried_this_step = False
            inference_ms = 0.0
            if pred_action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                queried_this_step = True
                # Use current-step text as prompt when multi-step decomposition is
                # available — the trajectory_overlay finetune was trained on atomic
                # per-step instructions ("pick up X", "place X in Y"), not multi-step.
                step_prompt = instruction
                cs = planning.current_step
                if cs is not None and isinstance(cs, dict):
                    s = cs.get("step")
                    if isinstance(s, str) and s.strip():
                        step_prompt = s.strip()
                req = {
                    "observation/exterior_image_1_left": model_input_ext,
                    "observation/wrist_image_left": model_input_wrist,
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": step_prompt,
                }
                t0 = time.time()
                with prevent_keyboard_interrupt():
                    pred_action_chunk = policy_client.infer(req)["actions"]
                inference_ms = (time.time() - t0) * 1000
                # Re-anchor integrator to actual measured q (matches main_robolab.py)
                q_target = curr_obs["joint_position"].copy()

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            # Server returns absolute joint POSITION targets (the
            # `pi05_droid_jointpos` config applies the AbsoluteActions
            # output transform; the trajectory_overlay finetune was
            # trained on top of that base). Use directly. Bang-bang
            # gripper at 0.5 to match RoboLab's Pi0DroidJointposClient.
            q_target = action[:7].astype(np.float32)
            # Threshold lowered from 0.5 to 0.3 to capture "hesitant" grasps where
            # the model peaks around 0.4 — successful runs jump from ~0 to 0.6+ in
            # one step, but some near-misses peak at 0.3-0.5 then drop.
            grip_cmd = 1.0 if float(action[7]) > 0.3 else 0.0

            # Build full action tensor (q_target + gripper) on the env device
            full_act = np.concatenate([q_target, np.array([grip_cmd], dtype=np.float32)])
            act_tensor = torch.as_tensor(full_act, dtype=torch.float32, device=env.device).unsqueeze(0)

            actions_log.write(
                f"t={t_step:04d} | queried={int(queried_this_step)} | "
                f"action={action.tolist()} | gripper={grip_cmd:.2f} | "
                f"chunk_idx={actions_from_chunk_completed}/{args.open_loop_horizon} | "
                f"inf_ms={inference_ms:.0f} | "
                f"q={curr_obs['joint_position'].tolist()}\n"
            )
            actions_log.flush()

            # Match main_robolab.py's tensor + multi-env handling.
            if env.num_envs > 1:
                act_tensor = act_tensor.expand(env.num_envs, -1).contiguous()
            obs, _, term, trunc, _ = env.step(act_tensor)
            if bool(term[0].item()):
                auto_success = True
                print(f"[overlay] env terminated (success) at t={t_step + 1}")
                break
            if bool(trunc[0].item()):
                auto_success = False
                print(f"[overlay] episode truncated at t={t_step + 1}")
                break
    except KeyboardInterrupt:
        print("[overlay] keyboard interrupt; stopping early")
    except Exception as e:
        print(f"[overlay] rollout exception: {e!r}")
        traceback.print_exc()
    finally:
        actions_log.close()

    # If rollout reached max_timesteps without termination, success is False.
    print(f"[overlay] auto_success = {auto_success}")
    with open(os.path.join(run_dir, "result.txt"), "w") as f:
        f.write(f"success: {'y' if auto_success else 'n'}\n")
        f.write(f"auto_success: {auto_success}\n")
        f.write(f"success_source: auto_detector\n")
        f.write(f"instruction: {instruction}\n")
        f.write(f"total_timesteps: {rollout_steps}\n")
        f.write(f"policy: pi05_droid_finetune (overlay, port {args.remote_port})\n")

    return 0 if auto_success else 1


# ---------------------------------------------------------------------------
# CLI / entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default")
    parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=["benchmark"])
    parser.add_argument("--max-timesteps", "--max_timesteps", type=int, default=300)
    parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=8)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--remote-host", "--remote_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote-port", "--remote_port", type=int, default=8002)
    parser.add_argument("--save-dir", "--save_dir", type=str, default="runs_overlay")
    parser.add_argument("--save-frames", "--save_frames", action="store_true")
    parser.add_argument("--plan-freq", "--plan_freq", type=int, default=150)
    parser.add_argument("--max-plan-count", "--max_plan_count", type=int, default=20)
    parser.add_argument("--gpt-model", "--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--gemini-model", "--gemini_model", type=str,
                        default="gemini-robotics-er-1.6-preview")
    parser.add_argument("--step-check-interval", "--step_check_interval", type=int, default=30,
                        help="Run Gemini step-completion check every N env steps. 0 disables.")
    parser.add_argument("--step-check-max-images", "--step_check_max_images", type=int, default=3,
                        help="Max recent frames sent to Gemini for completion check.")

    # AppLauncher args
    from isaaclab.app import AppLauncher  # noqa: E402
    AppLauncher.add_app_launcher_args(parser)
    ns, _ = parser.parse_known_args()
    ns.enable_cameras = True
    ns.headless = True

    launcher = AppLauncher(ns)
    app = launcher.app

    try:
        rc = main(Args(
            task=ns.task,
            instruction_type=ns.instruction_type,
            task_dirs=tuple(ns.task_dirs),
            max_timesteps=ns.max_timesteps,
            open_loop_horizon=ns.open_loop_horizon,
            num_envs=ns.num_envs,
            remote_host=ns.remote_host,
            remote_port=ns.remote_port,
            save_dir=ns.save_dir,
            save_frames=ns.save_frames,
            plan_freq=ns.plan_freq,
            max_plan_count=ns.max_plan_count,
            gpt_model=ns.gpt_model,
            gemini_model=ns.gemini_model,
            step_check_interval=ns.step_check_interval,
            step_check_max_images=ns.step_check_max_images,
        ), app)
    finally:
        try:
            app.close()
        except Exception:
            pass
    sys.exit(rc)
