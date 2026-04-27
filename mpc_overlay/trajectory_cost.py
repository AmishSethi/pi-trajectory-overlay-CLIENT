"""Differentiable trajectory cost for flow-matching guidance.

Given a normalized action chunk (joint velocities + gripper), integrate to joint
positions, run Franka forward kinematics, project to pixel space through the
exterior camera extrinsics+intrinsics, and compare against user-drawn waypoints
(in the 180-degree-flipped frame).

All tensor operations support autograd. No numpy/scipy inside this module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mpc_overlay.franka_fk import fk_ee_position


@dataclass
class GuidanceSpec:
    waypoints_px: Tensor          # (K, 2)  pixel (u,v) in the (possibly flipped) frame
    image_hw: tuple                # (H, W)  e.g. (720, 1280)
    K_intrinsics: Tensor          # (3, 3)  camera matrix at HD720
    extrinsic_cam_in_base: Tensor # (6,)    [x,y,z,rx,ry,rz]  meters + radians
    image_flipped_180: bool       # True for the exterior ZED client pipeline
    control_dt: float             # 1 / control_hz, e.g. 1/15
    joint_vel_scale: Tensor       # (7,)    rad/s per action=1
    q0: Tensor                    # (B, 7)  current 7-DoF joint angles
    waypoint_mask: Optional[Tensor] = None  # (K,) float; defaults to ones
    # Extra translation along the flange +Z axis to reach the actual gripper
    # tip (panda_hand_tcp on the Franka hand: 0.1034 m). 0.0 keeps the FK at
    # the flange (panda_link8), matching pre-existing unit tests.
    ee_offset_from_flange: float = 0.0
    # Sliding-window arrow target — if None, legacy "resample whole arrow" behaviour.
    # If set, the cost instead targets a sliding window ahead of the current EE:
    # chunk step k → arrow[s0 + k*(lookahead·total/(T-1))] where s0 is the current
    # EE's projected arc-length. Gives a true "follow-the-arrow" signal instead of
    # "ground-to-nearest-point". Typical value: 0.10–0.25 (fraction of arc length
    # per chunk). See sliding-window receding-horizon tracking / MPCC contouring
    # literature (Liniger et al. ICRA 2015) for the design origin.
    arrow_lookahead: Optional[float] = None
    # Reward for gripper closure when EE is near the arrow start, and for gripper
    # opening when near the arrow end. Addresses the observed grasp-stall failure
    # mode without overriding the VLA's overall grasp timing. 0.0 disables.
    # When enabled, `gripper_reward_weight` scales the additive bonus, and
    # `gripper_zone_frac` sets the arc-length thresholds (default 0.15 = first and
    # last 15%).
    gripper_reward_weight: float = 0.0
    gripper_zone_frac: float = 0.15
    # Post-CEM hard gripper override. When True, after CEM returns its best chunk,
    # if the current EE projects within `gripper_zone_frac` of the arrow start,
    # the gripper dim is forced to +1 (full close) across the chunk; similarly
    # if near the arrow end, forced to -1 (full open). This addresses the observed
    # "grasp stall" mode where the VLA closes the gripper momentarily and
    # immediately releases (e.g. mug-center, throw-apple runs) without the CEM
    # having any way to intervene -- the reward term above only helps when CEM
    # is also varying the gripper, which is gated off by freeze_gripper=True.
    gripper_force_override: bool = False
    # Current gripper position scalar in [0, 1] (0 = open, 1 = closed). Used by
    # gripper-state-aware arbitration: when the gripper is closed (carrying an
    # object), the MPC arrow target is for the *next* phase (placement) which
    # the VLA is better at — so MPC's arrow-pull is gated off via α_gripper
    # multiplied into the existing pixel-distance arbitration α.
    # None = old behaviour (no gripper-state arbitration).
    gripper_state_now: Optional[float] = None


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def _rx(theta: Tensor) -> Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    row0 = torch.stack([one, zero, zero], dim=-1)
    row1 = torch.stack([zero, c, -s], dim=-1)
    row2 = torch.stack([zero, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _ry(theta: Tensor) -> Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    row0 = torch.stack([c, zero, s], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-s, zero, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rz(theta: Tensor) -> Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def euler_xyz_to_matrix(rxryrz: Tensor) -> Tensor:
    """xyz intrinsic rotation matching scipy.Rotation.from_euler('xyz', r).

    scipy's 'xyz' (lowercase) is an *extrinsic* rotation about fixed X, Y, Z.
    The resulting matrix equals ``Rz(rz) @ Ry(ry) @ Rx(rx)`` -- the composition
    of intrinsic rotations in reverse order. Verified by the parity test.

    rxryrz: (..., 3)  -> (..., 3, 3)
    """
    rx, ry, rz = rxryrz[..., 0], rxryrz[..., 1], rxryrz[..., 2]
    Rx = _rx(rx)
    Ry = _ry(ry)
    Rz = _rz(rz)
    return Rz @ Ry @ Rx


def se3_inverse(R: Tensor, t: Tensor) -> tuple:
    """Closed-form SE(3) inverse. R: (..., 3, 3), t: (..., 3) -> (R^T, -R^T t)."""
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv


# --------------------------------------------------------------------------- #
# Pixel prediction
# --------------------------------------------------------------------------- #
def predict_ee_pixels(a: Tensor, spec: GuidanceSpec) -> Tensor:
    """a: (B, T, 8). Returns predicted EE pixel traj (B, T, 2) in the FLIPPED frame
    if spec.image_flipped_180 else in the raw frame.
    """
    B, T, _ = a.shape
    device = a.device
    dtype = a.dtype

    joint_vel_scale = spec.joint_vel_scale.to(device=device, dtype=dtype)
    q0 = spec.q0.to(device=device, dtype=dtype)              # (B, 7)
    dt = float(spec.control_dt)

    # Joint velocities -> delta-q per step, then cumulative sum -> q_k for k=1..T
    vel = a[..., :7] * joint_vel_scale                        # (B, T, 7)
    dq = vel * dt                                             # (B, T, 7)
    q_traj = q0.unsqueeze(1) + torch.cumsum(dq, dim=1)        # (B, T, 7)

    # FK: base-frame EE position. flange_offset is the panda_link7 -> panda_link8
    # distance (0.107 m; franka_fk default). Add any extra gripper-TCP offset on top
    # so the projected pixel lands on the actual fingertip, not the flange.
    flange_offset = 0.107 + float(spec.ee_offset_from_flange)
    p_base = fk_ee_position(q_traj, flange_offset=flange_offset)  # (B, T, 3)

    # Extrinsic: camera-in-base -> invert to base-in-camera
    ext = spec.extrinsic_cam_in_base.to(device=device, dtype=dtype)  # (6,)
    t_cb = ext[:3]                                            # camera origin in base
    R_cb = euler_xyz_to_matrix(ext[3:])                       # (3, 3) base<-cam
    R_bc, t_bc = se3_inverse(R_cb, t_cb)                      # cam<-base

    # Transform (B, T, 3) base-frame points into camera frame.
    # p_cam = R_bc @ p_base + t_bc
    p_cam = torch.einsum("ij,btj->bti", R_bc, p_base) + t_bc  # (B, T, 3)

    # Pinhole projection
    K = spec.K_intrinsics.to(device=device, dtype=dtype)      # (3, 3)
    uvw = torch.einsum("ij,btj->bti", K, p_cam)               # (B, T, 3)
    # Guard against exact zeros without masking autograd.
    w = uvw[..., 2:3]
    eps = torch.full_like(w, 1e-8)
    w_safe = torch.where(w.abs() < 1e-8, eps, w)
    uv_raw = uvw[..., :2] / w_safe                            # (B, T, 2)

    if spec.image_flipped_180:
        H, W = int(spec.image_hw[0]), int(spec.image_hw[1])
        u = (W - 1) - uv_raw[..., 0]
        v = (H - 1) - uv_raw[..., 1]
        uv = torch.stack([u, v], dim=-1)
    else:
        uv = uv_raw
    return uv


# --------------------------------------------------------------------------- #
# Progress-aware arrow helpers (v2)
# --------------------------------------------------------------------------- #
def _cumulative_arc_length(waypoints: Tensor) -> tuple:
    """Returns (cum_arc: (K,), total: scalar tensor). Handles K==1."""
    K = waypoints.shape[0]
    if K == 1:
        z = torch.zeros(1, dtype=waypoints.dtype, device=waypoints.device)
        return z, z.squeeze()
    seg = waypoints[1:] - waypoints[:-1]
    seg_len = torch.linalg.vector_norm(seg, dim=-1)
    cum = torch.cat([
        torch.zeros(1, dtype=waypoints.dtype, device=waypoints.device),
        torch.cumsum(seg_len, dim=0),
    ], dim=0)
    return cum, cum[-1]


def _project_ee_to_arc(ee_px: Tensor, waypoints: Tensor,
                       cum_arc: Tensor, total: Tensor) -> Tensor:
    """Project a single (2,) ``ee_px`` onto the piecewise-linear arrow and return the
    scalar arc length of the closest point. Uses per-segment perpendicular projection."""
    K = waypoints.shape[0]
    if K == 1 or float(total) <= 0.0:
        return torch.zeros((), dtype=waypoints.dtype, device=waypoints.device)
    seg = waypoints[1:] - waypoints[:-1]                   # (K-1, 2)
    seg_sqlen = (seg * seg).sum(dim=-1).clamp(min=1e-12)   # (K-1,)
    seg_len = seg_sqlen.sqrt()
    v = ee_px.unsqueeze(0) - waypoints[:-1]                # (K-1, 2)
    t_star = ((v * seg).sum(dim=-1) / seg_sqlen).clamp(0.0, 1.0)  # (K-1,)
    proj = waypoints[:-1] + t_star.unsqueeze(-1) * seg     # (K-1, 2)
    dist = torch.linalg.vector_norm(ee_px.unsqueeze(0) - proj, dim=-1)  # (K-1,)
    best = int(dist.argmin().item())
    return cum_arc[best] + t_star[best] * seg_len[best]


def _project_ee_to_arc_batch(ee_px: Tensor, waypoints: Tensor,
                             cum_arc: Tensor, total: Tensor) -> Tensor:
    """Batched version of _project_ee_to_arc.

    ee_px: (B, 2). Returns (B,) arc lengths. No ``.item()`` calls — stays on-device
    and differentiable w.r.t. ee_px (piecewise differentiable at segment boundaries).
    """
    K = waypoints.shape[0]
    B = ee_px.shape[0]
    device = ee_px.device
    dtype = ee_px.dtype
    if K == 1 or float(total) <= 0.0:
        return torch.zeros(B, dtype=dtype, device=device)
    seg = waypoints[1:] - waypoints[:-1]                   # (K-1, 2)
    seg_sqlen = (seg * seg).sum(dim=-1).clamp(min=1e-12)   # (K-1,)
    seg_len = seg_sqlen.sqrt()                             # (K-1,)
    # Vector from segment start to each ee_px point: (B, K-1, 2)
    v = ee_px.unsqueeze(1) - waypoints[:-1].unsqueeze(0)
    # Dot each v with its segment, divide by seg_sqlen → t_star in [0, 1] per segment.
    t_star = ((v * seg.unsqueeze(0)).sum(dim=-1) / seg_sqlen.unsqueeze(0)).clamp(0.0, 1.0)  # (B, K-1)
    proj = waypoints[:-1].unsqueeze(0) + t_star.unsqueeze(-1) * seg.unsqueeze(0)  # (B, K-1, 2)
    dist = torch.linalg.vector_norm(ee_px.unsqueeze(1) - proj, dim=-1)  # (B, K-1)
    best = dist.argmin(dim=-1)                             # (B,)
    t_star_best = t_star.gather(1, best.unsqueeze(-1)).squeeze(-1)  # (B,)
    seg_len_best = seg_len[best]                           # (B,)
    cum_arc_best = cum_arc[best]                           # (B,)
    return cum_arc_best + t_star_best * seg_len_best


def _sample_arrow_segment(waypoints: Tensor, cum_arc: Tensor, total: Tensor,
                          s_start: Tensor, s_end: Tensor, T: int) -> Tensor:
    """Sample T points on the arrow, linearly in arc length from s_start to s_end.
    s_start and s_end are scalar tensors in [0, total]. Returns (T, 2)."""
    K = waypoints.shape[0]
    if K == 1 or float(total) <= 0.0:
        return waypoints[0:1].expand(T, 2).clone()
    # linspace s in absolute arc-length units
    alpha = torch.linspace(0.0, 1.0, T,
                           dtype=waypoints.dtype, device=waypoints.device)
    s_q = s_start + alpha * (s_end - s_start)
    s_q = s_q.clamp(min=0.0, max=float(total))
    idx = torch.searchsorted(cum_arc, s_q, right=False)
    idx = torch.clamp(idx, min=1, max=K - 1)
    j0 = idx - 1
    j1 = idx
    s0_arr = cum_arc[j0]
    s1_arr = cum_arc[j1]
    denom = (s1_arr - s0_arr).clamp(min=1e-12)
    t = ((s_q - s0_arr) / denom).clamp(0.0, 1.0).unsqueeze(-1)
    p0 = waypoints[j0]
    p1 = waypoints[j1]
    return p0 + t * (p1 - p0)


def ee_pixel_at_q0(spec: GuidanceSpec, device=None, dtype=None) -> Tensor:
    """Return the (2,) pixel of the EE corresponding to spec.q0 (first batch row)."""
    if device is None:
        device = spec.q0.device
    if dtype is None:
        dtype = spec.q0.dtype
    q0 = spec.q0
    q0_1 = q0.unsqueeze(0) if q0.dim() == 1 else q0[:1]
    spec_1 = dataclasses.replace(spec, q0=q0_1)
    fake = torch.zeros((1, 1, 8), device=device, dtype=dtype)
    return predict_ee_pixels(fake, spec_1)[0, 0]


def build_arrow_target(spec: GuidanceSpec, T: int,
                       ee_px_now: Optional[Tensor] = None,
                       device=None, dtype=None) -> Tensor:
    """Return (T, 2) target waypoints.

    If ``spec.arrow_lookahead`` is None: legacy behaviour (T evenly-arc-length-
    spaced points along the whole arrow).

    Otherwise: a sliding window — T points on arc length ``[s0, s0+lookahead*total]``
    where s0 is the current EE's projected arc length. This gives CEM a reachable
    per-step target that ADVANCES through the arrow instead of a static full-arrow
    target that degenerates to "ground to nearest point".
    """
    if device is None:
        device = spec.waypoints_px.device
    if dtype is None:
        dtype = spec.waypoints_px.dtype
    waypoints = spec.waypoints_px.detach().to(device=device, dtype=dtype)

    if spec.arrow_lookahead is None:
        return _resample_waypoints(waypoints, T)

    cum_arc, total = _cumulative_arc_length(waypoints)
    if float(total) <= 0.0:
        return waypoints[0:1].expand(T, 2).clone()

    if ee_px_now is None:
        ee_px_now = ee_pixel_at_q0(spec, device=device, dtype=dtype)

    s0 = _project_ee_to_arc(ee_px_now.to(device=device, dtype=dtype),
                            waypoints, cum_arc, total)
    lookahead = float(spec.arrow_lookahead)
    s_end = (s0 + lookahead * total).clamp(max=float(total))
    return _sample_arrow_segment(waypoints, cum_arc, total, s0, s_end, T)


# --------------------------------------------------------------------------- #
# Waypoint arc-length resampling (legacy full-arrow resampling)
# --------------------------------------------------------------------------- #
def _resample_waypoints(waypoints: Tensor, T: int) -> Tensor:
    """waypoints: (K, 2). Returns (T, 2), piecewise-linear along cumulative arc length.

    waypoints are detached constants; gradients do not flow back through this.
    """
    K = waypoints.shape[0]
    wp = waypoints.detach()
    if K == 1:
        return wp.expand(T, 2).clone()

    seg = wp[1:] - wp[:-1]                                    # (K-1, 2)
    seg_len = torch.linalg.vector_norm(seg, dim=-1)           # (K-1,)
    cum = torch.cat([torch.zeros(1, dtype=wp.dtype, device=wp.device),
                     torch.cumsum(seg_len, dim=0)], dim=0)    # (K,)
    total = cum[-1]
    if float(total) <= 0.0:
        return wp[0:1].expand(T, 2).clone()
    s = cum / total                                           # (K,) in [0,1]

    t_grid = torch.linspace(0.0, 1.0, T, dtype=wp.dtype, device=wp.device)

    # For each t in t_grid, find the segment [s[j], s[j+1]] containing it.
    # searchsorted gives the insertion index; clamp to [1, K-1].
    idx = torch.searchsorted(s, t_grid, right=False)
    idx = torch.clamp(idx, min=1, max=K - 1)
    j0 = idx - 1
    j1 = idx
    s0 = s[j0]
    s1 = s[j1]
    denom = (s1 - s0).clamp(min=1e-12)
    alpha = ((t_grid - s0) / denom).clamp(0.0, 1.0).unsqueeze(-1)
    p0 = wp[j0]
    p1 = wp[j1]
    return p0 + alpha * (p1 - p0)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def trajectory_cost(a: Tensor, spec: GuidanceSpec) -> Tensor:
    """Scalar cost: masked L2 between predicted EE pixel trajectory and a target
    waypoint trajectory. Two modes, depending on ``spec.arrow_lookahead``:
      - None (legacy): target = full arrow arc-length-resampled to T points.
      - float: target = sliding window of length lookahead·total starting at the
        current EE projection — gives a genuine "advance along arrow" signal.
    """
    pred = predict_ee_pixels(a, spec)                         # (B, T, 2)
    B, T, _ = pred.shape

    target = build_arrow_target(spec, T, device=pred.device, dtype=pred.dtype)
    # (T, 2) -> broadcast to (B, T, 2)
    target = target.unsqueeze(0).expand(B, T, 2)

    if spec.waypoint_mask is None:
        mask = torch.ones(T, dtype=pred.dtype, device=pred.device)
    else:
        wm = spec.waypoint_mask.to(device=pred.device, dtype=pred.dtype)
        # Resample the mask to length T the same way (nearest in arc-length space).
        if wm.shape[0] == T:
            mask = wm
        else:
            # Use the same arc-length grid; take the mask from the nearer endpoint.
            K = spec.waypoints_px.shape[0]
            if K == 1:
                mask = wm[0:1].expand(T).clone()
            else:
                wp = spec.waypoints_px.detach().to(device=pred.device, dtype=pred.dtype)
                seg = wp[1:] - wp[:-1]
                seg_len = torch.linalg.vector_norm(seg, dim=-1)
                cum = torch.cat([
                    torch.zeros(1, dtype=pred.dtype, device=pred.device),
                    torch.cumsum(seg_len, dim=0),
                ], dim=0)
                total = cum[-1].clamp(min=1e-12)
                s = cum / total
                t_grid = torch.linspace(0.0, 1.0, T, dtype=pred.dtype, device=pred.device)
                idx = torch.searchsorted(s, t_grid, right=False)
                idx = torch.clamp(idx, min=1, max=K - 1)
                j0 = idx - 1
                j1 = idx
                alpha = ((t_grid - s[j0]) / (s[j1] - s[j0]).clamp(min=1e-12)).clamp(0.0, 1.0)
                # Nearest-neighbour mask: round alpha.
                nearest = torch.where(alpha < 0.5, j0, j1)
                mask = wm[nearest]

    diff_sq = (pred - target).pow(2).sum(dim=-1)              # (B, T)
    weighted = diff_sq * mask.unsqueeze(0)                    # (B, T)
    mask_sum = mask.sum().clamp(min=1e-12)
    cost = weighted.sum(dim=-1) / mask_sum                    # (B,)
    return cost.mean()
