"""Tests for trajectory_cost.py.

Covers:
  1. Euler-parity vs scipy on 20 random rotations
  2. Identity-projection with a camera at (0,0,1) looking along -z
  3. 180-degree flip parity
  4. Zero-cost construction
  5. Finite-difference autograd check
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mpc_overlay import franka_fk
from mpc_overlay.trajectory_cost import (
    GuidanceSpec,
    euler_xyz_to_matrix,
    predict_ee_pixels,
    se3_inverse,
    trajectory_cost,
)


# --------------------------------------------------------------------------- #
# 1. Euler-parity
# --------------------------------------------------------------------------- #
def test_euler_xyz_matches_scipy():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(20):
        r = rng.uniform(-math.pi, math.pi, size=(3,))
        scipy_R = Rotation.from_euler("xyz", r).as_matrix()
        torch_R = euler_xyz_to_matrix(torch.tensor(r, dtype=torch.float64)).numpy()
        err = float(np.max(np.abs(scipy_R - torch_R)))
        max_err = max(max_err, err)
    assert max_err <= 1e-6, f"Euler parity mismatch: max abs err = {max_err:e}"


# --------------------------------------------------------------------------- #
# 2. Identity-projection
# --------------------------------------------------------------------------- #
def test_identity_projection():
    """Camera at (0,0,1) with identity rotation (no 180-deg flip for this test).

    With identity rotation, the camera frame coincides with the base frame axes,
    shifted by +z = 1 meter. For a base-frame point p, camera coords are
    p_cam = p - (0, 0, 1). With K = I the pixel is (p_cam_x / p_cam_z,
    p_cam_y / p_cam_z).
    """
    dtype = torch.float64
    # Put the Panda at its home-ish pose and get the actual EE position.
    q0 = torch.zeros(1, 7, dtype=dtype)
    ee = franka_fk.fk_ee_position(q0)[0]                     # (3,)
    # Make the camera sit 1 meter above the EE along +z, pointed along -z.
    cam_x, cam_y, cam_z = float(ee[0]), float(ee[1]), float(ee[2]) + 1.0

    spec = GuidanceSpec(
        waypoints_px=torch.zeros(2, 2, dtype=dtype),
        image_hw=(720, 1280),
        K_intrinsics=torch.eye(3, dtype=dtype),
        extrinsic_cam_in_base=torch.tensor([cam_x, cam_y, cam_z, 0.0, 0.0, 0.0], dtype=dtype),
        image_flipped_180=False,
        control_dt=1.0 / 15.0,
        joint_vel_scale=torch.ones(7, dtype=dtype),
        q0=q0,
    )
    # Zero velocity => EE stays at its initial position for every timestep.
    a = torch.zeros(1, 3, 8, dtype=dtype)
    uv = predict_ee_pixels(a, spec)                           # (1, 3, 2)

    # Hand-computed: p_cam = (0, 0, -1), so u = 0 / -1 = 0, v = 0 / -1 = 0.
    expected = torch.zeros(1, 3, 2, dtype=dtype)
    assert torch.allclose(uv, expected, atol=1e-9), f"{uv} != {expected}"


# --------------------------------------------------------------------------- #
# 3. Flip parity
# --------------------------------------------------------------------------- #
def test_flip_parity():
    """Raw pixel (10, 20) with image_hw=(720, 1280) must flip to (1269, 699)."""
    dtype = torch.float64
    q0 = torch.zeros(1, 7, dtype=dtype)
    ee = franka_fk.fk_ee_position(q0)[0]

    # Camera at ee + (0, 0, 1), identity rot, so p_cam = (0, 0, -1) at ee.
    # Pinhole K @ p_cam = (-cx, -cy, -1); divide by w = -1 -> (cx, cy). So to
    # land at raw pixel (10, 20) set cx = 10, cy = 20.
    K = torch.tensor([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]], dtype=dtype)

    cam_z = float(ee[2]) + 1.0
    spec_raw = GuidanceSpec(
        waypoints_px=torch.zeros(2, 2, dtype=dtype),
        image_hw=(720, 1280),
        K_intrinsics=K,
        extrinsic_cam_in_base=torch.tensor(
            [float(ee[0]), float(ee[1]), cam_z, 0.0, 0.0, 0.0], dtype=dtype
        ),
        image_flipped_180=False,
        control_dt=1.0 / 15.0,
        joint_vel_scale=torch.ones(7, dtype=dtype),
        q0=q0,
    )
    a = torch.zeros(1, 1, 8, dtype=dtype)
    uv_raw = predict_ee_pixels(a, spec_raw)[0, 0]
    assert torch.allclose(uv_raw, torch.tensor([10.0, 20.0], dtype=dtype), atol=1e-9), uv_raw

    spec_flip = GuidanceSpec(**{**spec_raw.__dict__, "image_flipped_180": True})
    uv_flip = predict_ee_pixels(a, spec_flip)[0, 0]
    expected = torch.tensor([1280 - 1 - 10.0, 720 - 1 - 20.0], dtype=dtype)  # (1269, 699)
    assert torch.allclose(uv_flip, expected, atol=1e-9), f"{uv_flip} != {expected}"


# --------------------------------------------------------------------------- #
# 4. Zero-cost test
# --------------------------------------------------------------------------- #
def test_zero_cost_when_waypoints_match_prediction():
    """With zero actions the EE stays still. If the waypoint polyline is the
    (constant) predicted pixel trajectory, cost must be ~0.
    """
    dtype = torch.float64
    q0 = torch.zeros(1, 7, dtype=dtype)
    ee = franka_fk.fk_ee_position(q0)[0]
    cam_z = float(ee[2]) + 1.0
    K = torch.eye(3, dtype=dtype)
    spec_probe = GuidanceSpec(
        waypoints_px=torch.zeros(2, 2, dtype=dtype),          # placeholder
        image_hw=(720, 1280),
        K_intrinsics=K,
        extrinsic_cam_in_base=torch.tensor(
            [float(ee[0]), float(ee[1]), cam_z, 0.0, 0.0, 0.0], dtype=dtype
        ),
        image_flipped_180=True,
        control_dt=1.0 / 15.0,
        joint_vel_scale=torch.ones(7, dtype=dtype),
        q0=q0,
    )
    T = 15
    a = torch.zeros(1, T, 8, dtype=dtype)
    pred_flip = predict_ee_pixels(a, spec_probe)[0]           # (T, 2)

    # Use the predicted traj as waypoints (K = T) so arc-length resampling keeps it identical.
    spec = GuidanceSpec(**{**spec_probe.__dict__, "waypoints_px": pred_flip.clone()})
    cost = trajectory_cost(a, spec)
    assert float(cost) <= 1e-4, f"cost = {float(cost)}"


# --------------------------------------------------------------------------- #
# 5. Autograd / finite-difference
# --------------------------------------------------------------------------- #
def test_autograd_finite_difference():
    dtype = torch.float64
    torch.manual_seed(1)
    B, T = 1, 6
    q0 = torch.zeros(B, 7, dtype=dtype)
    ee = franka_fk.fk_ee_position(q0)[0]
    cam_z = float(ee[2]) + 1.0
    spec = GuidanceSpec(
        # random-ish waypoints near the origin pixel
        waypoints_px=torch.tensor(
            [[0.1, 0.05], [0.2, 0.0], [0.25, -0.1], [0.3, -0.15]], dtype=dtype
        ),
        image_hw=(720, 1280),
        K_intrinsics=torch.eye(3, dtype=dtype),
        extrinsic_cam_in_base=torch.tensor(
            [float(ee[0]), float(ee[1]), cam_z, 0.0, 0.0, 0.0], dtype=dtype
        ),
        image_flipped_180=True,
        control_dt=1.0 / 15.0,
        joint_vel_scale=torch.full((7,), 0.1, dtype=dtype),
        q0=q0,
    )

    a = (0.01 * torch.randn(B, T, 8, dtype=dtype)).requires_grad_(True)
    cost = trajectory_cost(a, spec)
    cost.backward()
    ana = a.grad.detach().clone()

    # Finite differences on the first 7 dims (cost ignores dim 7).
    eps = 1e-6
    num = torch.zeros_like(ana)
    with torch.no_grad():
        for b in range(B):
            for t in range(T):
                for d in range(7):
                    a_p = a.detach().clone()
                    a_m = a.detach().clone()
                    a_p[b, t, d] += eps
                    a_m[b, t, d] -= eps
                    c_p = trajectory_cost(a_p, spec)
                    c_m = trajectory_cost(a_m, spec)
                    num[b, t, d] = (c_p - c_m) / (2 * eps)

    err = float((ana[..., :7] - num[..., :7]).abs().max())
    assert err < 1e-3, f"autograd mismatch, max abs err = {err:e}"


# --------------------------------------------------------------------------- #
# SE(3) inverse sanity (extra, cheap)
# --------------------------------------------------------------------------- #
def test_se3_inverse_roundtrip():
    torch.manual_seed(0)
    r = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    R = euler_xyz_to_matrix(r)
    t = torch.tensor([0.2, -0.5, 1.3], dtype=torch.float64)
    R_inv, t_inv = se3_inverse(R, t)
    # Applying then un-applying should be identity.
    p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    q = R @ p + t
    p_back = R_inv @ q + t_inv
    assert torch.allclose(p, p_back, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
