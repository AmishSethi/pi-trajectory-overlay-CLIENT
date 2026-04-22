import math

import numpy as np
import pytest
import torch

from mpc_overlay.franka_fk import fk_ee_pose, fk_ee_position


def test_home_position():
    # With q=0 and the Franka spec modified-DH params + 0.107 m flange offset,
    # the flange origin sits at roughly (0.088, 0, 0.926) in the base frame.
    q = torch.zeros(7, dtype=torch.float64)
    p = fk_ee_position(q)
    expected = torch.tensor([0.088, 0.0, 0.926], dtype=torch.float64)
    assert torch.allclose(p, expected, atol=1e-3), f"home position off: {p}"


def test_joint1_rotation():
    # Rotating joint 1 by +pi/2 rotates the whole arm about base z-axis.
    # So (x, y) at home maps to (-y_home, x_home) = (0, 0.088) and z unchanged.
    q_home = torch.zeros(7, dtype=torch.float64)
    q_rot = q_home.clone()
    q_rot[0] = math.pi / 2

    p_home = fk_ee_position(q_home)
    p_rot = fk_ee_position(q_rot)

    expected = torch.tensor([-p_home[1], p_home[0], p_home[2]], dtype=torch.float64)
    assert torch.allclose(p_rot, expected, atol=1e-6), f"rotated: {p_rot}, expected {expected}"


def test_autograd_gradient_matches_finite_difference():
    torch.manual_seed(0)
    q = torch.randn(7, dtype=torch.float64, requires_grad=True)

    # Autograd gradient of x-component.
    p = fk_ee_position(q)
    px = p[0]
    (g_auto,) = torch.autograd.grad(px, q)
    assert torch.isfinite(g_auto).all()

    # Central finite-difference reference.
    eps = 1e-4
    g_fd = torch.zeros(7, dtype=torch.float64)
    with torch.no_grad():
        for i in range(7):
            qp = q.detach().clone()
            qm = q.detach().clone()
            qp[i] += eps
            qm[i] -= eps
            g_fd[i] = (fk_ee_position(qp)[0] - fk_ee_position(qm)[0]) / (2 * eps)

    max_err = (g_auto - g_fd).abs().max().item()
    assert max_err <= 1e-3, f"grad mismatch, max err {max_err}"


def test_batched_shapes():
    q = torch.zeros(3, 5, 7, dtype=torch.float32)
    assert fk_ee_position(q).shape == (3, 5, 3)
    assert fk_ee_pose(q).shape == (3, 5, 4, 4)


def test_pose_is_proper_rigid_transform():
    torch.manual_seed(1)
    q = torch.randn(4, 7, dtype=torch.float64)
    T = fk_ee_pose(q)
    R = T[..., :3, :3]
    # Orthonormal rotation: R^T R = I, det(R) = +1.
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-6)
    assert torch.allclose(torch.linalg.det(R), torch.ones(4, dtype=torch.float64), atol=1e-6)
    # Bottom row is [0, 0, 0, 1].
    assert torch.allclose(T[..., 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(4, 4))
