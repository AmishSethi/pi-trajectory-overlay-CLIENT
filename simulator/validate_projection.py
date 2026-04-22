"""Visual sanity check for the sim extrinsic + intrinsic + FK pipeline.

Boots Isaac Sim, creates a RoboLab task, resets twice, grabs ONE external_cam
frame + the current 7-DoF joint positions, then:

  1. Computes the K matrix from our intrinsic formula.
  2. Queries the live extrinsic (composed against robot root pose).
  3. Projects several candidate action chunks through ``predict_ee_pixels``
     and overlays the resulting pixel trajectories on the captured frame.

Outputs:
  /tmp/sim_projection_check.png — one annotated frame, red dot = predicted EE
    at q0 (zero-action chunk), short green curve = joint-1 perturbation,
    short blue curve = joint-2 perturbation.

All three overlays should sit roughly on the gripper; the green/blue curves
should trace small coherent arcs (not zigzags or off-screen excursions).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLIENT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _CLIENT_ROOT not in sys.path:
    sys.path.insert(0, _CLIENT_ROOT)


def main(task: str, save_path: str):
    from isaaclab.app import AppLauncher  # type: ignore  # noqa: E402

    # --- Isaac Sim boot (must precede any isaaclab / robolab import) -----
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    ns, _ = parser.parse_known_args([])
    ns.enable_cameras = True
    ns.headless = True
    launcher = AppLauncher(ns)
    app = launcher.app

    try:
        import cv2  # noqa: F401  — must be imported before isaaclab
        import torch

        from robolab.core.environments.runtime import create_env
        from robolab.core.logging.recorder_manager import patch_recorder_manager
        from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs

        from mpc_overlay import GuidanceSpec, predict_ee_pixels, fk_ee_position
        from simulator.mpc_sim_adapter import (
            build_sim_guidance_spec,
            compute_sim_intrinsics,
            get_sim_extrinsic_6vec,
        )

        patch_recorder_manager()
        auto_register_droid_envs(task_dirs=["benchmark"], task=task)
        env, _ = create_env(task, num_envs=1, use_fabric=True)

        obs, _ = env.reset()
        obs, _ = env.reset()

        # --- pull sim state -------------------------------------------------
        ext_img = np.asarray(
            obs["image_obs"]["external_cam"][0].detach().cpu().numpy(), dtype=np.uint8,
        )
        q0 = np.asarray(
            obs["proprio_obs"]["arm_joint_pos"][0].detach().cpu().numpy(), dtype=np.float32,
        )
        cam = env.scene.sensors["external_cam"]
        cam_pos_w = cam.data.pos_w[0].cpu().numpy()
        cam_quat_w_ros = cam.data.quat_w_ros[0].cpu().numpy()
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
        root_quat_w = robot.data.root_quat_w[0].cpu().numpy()

        print(f"[sim] image shape = {ext_img.shape}  dtype = {ext_img.dtype}")
        print(f"[sim] q0 (7 joints, rad) = {np.round(q0, 3).tolist()}")
        print(f"[sim] cam_pos_w     = {np.round(cam_pos_w, 3).tolist()}")
        print(f"[sim] cam_quat_w_ros = {np.round(cam_quat_w_ros, 3).tolist()}  (assumed wxyz)")
        print(f"[sim] root_pos_w    = {np.round(root_pos_w, 3).tolist()}")
        print(f"[sim] root_quat_w   = {np.round(root_quat_w, 3).tolist()}  (assumed wxyz)")

        K = compute_sim_intrinsics()
        ext6 = get_sim_extrinsic_6vec(env)
        print(f"[mpc] K diag fx,fy = ({float(K[0,0]):.2f}, {float(K[1,1]):.2f}), cx,cy = ({float(K[0,2]):.2f}, {float(K[1,2]):.2f})")
        print(f"[mpc] extrinsic 6-vec (cam_in_base) [x,y,z,rx,ry,rz] = {np.round(ext6.numpy(), 3).tolist()}")

        # --- build spec with a dummy waypoint set; we'll use predict_ee_pixels directly ---
        dummy_wp = np.array([[640, 360], [640, 360]], dtype=np.float32)
        spec = build_sim_guidance_spec(env, waypoints_px=dummy_wp, q0_7=q0[:7])

        # Also confirm the pure-FK EE position (no projection yet) — tells us where the
        # gripper sits in robot base frame, useful for debugging.
        q0_t = torch.as_tensor(q0[:7], dtype=torch.float32)
        ee_base = fk_ee_position(q0_t).numpy()
        print(f"[fk]  EE in robot base frame (m)     = {np.round(ee_base, 3).tolist()}")

        T = 15
        chunks = {
            "zero":     (torch.zeros((1, T, 8), dtype=torch.float32),                           "red"),
            "joint1+":  (_chunk_with_constant(T, dim=0, value=0.15),                            "lime"),
            "joint2+":  (_chunk_with_constant(T, dim=1, value=0.15),                            "cyan"),
        }

        overlay = Image.fromarray(ext_img).convert("RGB")
        draw = ImageDraw.Draw(overlay)

        for label, (a, color) in chunks.items():
            pred_px = predict_ee_pixels(a, spec)[0].numpy()  # (T, 2) in raw image pixels
            u0, v0 = float(pred_px[0, 0]), float(pred_px[0, 1])
            uT, vT = float(pred_px[-1, 0]), float(pred_px[-1, 1])
            print(f"[proj] {label:10s}  start=({u0:7.1f},{v0:7.1f})   end=({uT:7.1f},{vT:7.1f})")
            prev = None
            for pt in pred_px:
                u, v = float(pt[0]), float(pt[1])
                draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color, outline=color)
                if prev is not None:
                    draw.line([prev, (u, v)], fill=color, width=2)
                prev = (u, v)
            draw.ellipse([u0 - 7, v0 - 7, u0 + 7, v0 + 7], outline=color, width=2)
            draw.text((u0 + 10, v0 - 8), label, fill=color)

        overlay.save(save_path)
        print(f"[out] wrote {save_path}  size={ext_img.shape[1]}x{ext_img.shape[0]}")
    finally:
        app.close()


def _chunk_with_constant(T: int, dim: int, value: float):
    import torch
    a = torch.zeros((1, T, 8), dtype=torch.float32)
    a[0, :, dim] = value
    return a


if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="BananaOnPlateTask")
    p.add_argument("--save-path", default="/tmp/sim_projection_check.png")
    # Let AppLauncher args be parsed inside main()
    known, _ = p.parse_known_args()
    main(task=known.task, save_path=known.save_path)
