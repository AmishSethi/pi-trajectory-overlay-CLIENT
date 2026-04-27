"""Sim-specific adapter: builds an ``mpc_overlay.GuidanceSpec`` from RoboLab env state.

The RoboLab tasks we target (e.g. BananaOnPlateTask) mount a single exterior camera
``external_cam`` rendered at 1280x720 in OpenGL/+Y-up convention. Isaac Sim does not
expose a 3x3 K matrix at runtime, so we compute intrinsics from the camera config
constants baked into ``OverShoulderLeftCameraCfg``. Extrinsics ARE queryable at
runtime (``sensor.data.pos_w`` + ``quat_w_ros``) so we compose them with the robot's
root world pose at every call — this is cheap (four CPU matmuls) and keeps the spec
fresh even if the robot base moves.

Everything here is **sim-specific** — the real-robot client stays on factory ZED
intrinsics + stored calibration.json extrinsics, implemented separately.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mpc_overlay import GuidanceSpec

# --- External camera config (from RoboLab's OverShoulderLeftCameraCfg) ------
# third_party/RoboLab/robolab/variations/camera.py :: OverShoulderLeftCameraCfg
EXT_CAM_FOCAL_LENGTH = 2.1     # units: same as apertures — cancel out in the pinhole formula
EXT_CAM_HORIZ_APERTURE = 5.376
EXT_CAM_VERT_APERTURE = 3.024
EXT_CAM_H = 720
EXT_CAM_W = 1280

# --- DROID joint-space control constants (mirror main_robolab.py) -----------
DROID_VEL_LIMITS = torch.tensor(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=torch.float32,
)
DROID_CONTROL_HZ = 15.0


def compute_sim_intrinsics(
    focal_length: float = EXT_CAM_FOCAL_LENGTH,
    horiz_aperture: float = EXT_CAM_HORIZ_APERTURE,
    vert_aperture: float = EXT_CAM_VERT_APERTURE,
    H: int = EXT_CAM_H,
    W: int = EXT_CAM_W,
) -> torch.Tensor:
    """Compute a pinhole K matrix (3, 3) from focal length + sensor aperture + image size.

    ``fx = focal_length * W / horiz_aperture``; ``fy`` analogous. The focal length and
    aperture must be in the same units (mm or cm); Isaac Sim's default is cm.
    Principal point defaults to the image center.
    """
    fx = focal_length * W / horiz_aperture
    fy = focal_length * H / vert_aperture
    cx = W / 2.0
    cy = H / 2.0
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _quat_wxyz_to_scipy(q_wxyz: np.ndarray) -> R:
    """IsaacLab stores quaternions as (w, x, y, z); scipy wants (x, y, z, w)."""
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def get_sim_extrinsic_6vec(
    env: Any,
    camera_name: str = "external_cam",
    robot_name: str = "robot",
    env_id: int = 0,
) -> torch.Tensor:
    """Return the camera's pose in the robot base frame as a 6-vec [x,y,z,rx,ry,rz].

    Euler rotation is scipy's intrinsic-axis ``'xyz'`` convention (radians), matching
    what ``trajectory_cost.euler_xyz_to_matrix`` expects (verified in the existing
    trajectory_cost tests vs scipy parity to 1e-6).

    Composition:  ``T_base<-cam = T_world<-base^{-1} @ T_world<-cam``.
    Rotation:      ``R_base<-cam = R_world<-base^{-1} * R_world<-cam``.
    Position:      ``p_base<-cam = R_world<-base^{-1} @ (p_world_cam - p_world_base)``.
    """
    cam = env.scene.sensors[camera_name]
    cam_pos_w = cam.data.pos_w[env_id].detach().cpu().numpy()            # (3,)
    cam_quat_w = cam.data.quat_w_ros[env_id].detach().cpu().numpy()      # (4,) wxyz

    robot = env.scene[robot_name]
    base_pos_w = robot.data.root_pos_w[env_id].detach().cpu().numpy()    # (3,)
    base_quat_w = robot.data.root_quat_w[env_id].detach().cpu().numpy()  # (4,) wxyz

    cam_R_w = _quat_wxyz_to_scipy(cam_quat_w)
    base_R_w = _quat_wxyz_to_scipy(base_quat_w)

    base_R_w_inv = base_R_w.inv()
    p_base_cam = base_R_w_inv.apply(cam_pos_w - base_pos_w)
    R_base_cam = base_R_w_inv * cam_R_w
    euler_xyz = R_base_cam.as_euler("xyz")

    return torch.tensor(
        np.concatenate([p_base_cam, euler_xyz]).astype(np.float32),
        dtype=torch.float32,
    )


def build_sim_guidance_spec(
    env: Any,
    waypoints_px: np.ndarray,
    q0_7: np.ndarray,
    *,
    camera_name: str = "external_cam",
    robot_name: str = "robot",
    env_id: int = 0,
    control_hz: float = DROID_CONTROL_HZ,
    joint_vel_scale: np.ndarray | torch.Tensor | None = None,
    waypoint_mask: np.ndarray | torch.Tensor | None = None,
    K_intrinsics: torch.Tensor | None = None,
    extrinsic_override: torch.Tensor | None = None,
    ee_offset_from_flange: float = 0.1034,  # Franka Hand TCP (panda_hand_tcp vs panda_link8)
    arrow_lookahead: float | None = None,
    gripper_reward_weight: float = 0.0,
    gripper_zone_frac: float = 0.15,
    gripper_force_override: bool = False,
    gripper_state_now: float | None = None,
) -> GuidanceSpec:
    """Assemble a ``GuidanceSpec`` ready to pass to ``mpc_overlay(...)``.

    ``waypoints_px`` must live in the **raw 1280x720 image frame** (no flip — sim
    renders right-side up). ``q0_7`` is the current 7-DoF Franka joint positions in
    radians. ``extrinsic_override`` exists for unit tests; production callers pass
    ``None`` and let us query it from ``env``.
    """
    K = K_intrinsics if K_intrinsics is not None else compute_sim_intrinsics()
    ext6 = (
        extrinsic_override
        if extrinsic_override is not None
        else get_sim_extrinsic_6vec(env, camera_name=camera_name, robot_name=robot_name, env_id=env_id)
    )

    wp = torch.as_tensor(np.asarray(waypoints_px), dtype=torch.float32)
    if wp.ndim != 2 or wp.shape[-1] != 2:
        raise ValueError(f"waypoints_px must be (K, 2), got {tuple(wp.shape)}")

    q0 = torch.as_tensor(np.asarray(q0_7)[:7], dtype=torch.float32).unsqueeze(0)  # (1, 7)

    if joint_vel_scale is None:
        jvs = DROID_VEL_LIMITS.clone()
    else:
        jvs = torch.as_tensor(np.asarray(joint_vel_scale), dtype=torch.float32)

    wm = None
    if waypoint_mask is not None:
        wm = torch.as_tensor(np.asarray(waypoint_mask), dtype=torch.float32)

    return GuidanceSpec(
        waypoints_px=wp,
        image_hw=(EXT_CAM_H, EXT_CAM_W),
        K_intrinsics=K,
        extrinsic_cam_in_base=ext6,
        image_flipped_180=False,  # sim renders right-side up
        control_dt=1.0 / control_hz,
        joint_vel_scale=jvs,
        q0=q0,
        waypoint_mask=wm,
        ee_offset_from_flange=float(ee_offset_from_flange),
        arrow_lookahead=(None if arrow_lookahead is None else float(arrow_lookahead)),
        gripper_reward_weight=float(gripper_reward_weight),
        gripper_zone_frac=float(gripper_zone_frac),
        gripper_force_override=bool(gripper_force_override),
        gripper_state_now=(None if gripper_state_now is None else float(gripper_state_now)),
    )
