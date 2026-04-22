"""Differentiable forward kinematics for the Franka Emika Panda arm.

Modified-DH (Craig 1989) convention:
    T_i = Rot_x(alpha_{i-1}) * Trans_x(a_{i-1}) * Rot_z(theta_i) * Trans_z(d_i)

DH parameters are from Franka Emika's "Robot and Interface Specification"
(https://frankaemika.github.io/docs/control_parameters.html, section
"Denavit-Hartenberg parameters"). Cross-checked against the URDF in
frankaemika/franka_ros -> franka_description/robots/common/franka_robot.xacro
(same a, d, alpha values; flange link 'panda_link8' is offset 0.107 m along
z from 'panda_link7').

The flange origin (panda_link8) is used as the EE reference point. A gripper
offset is not applied here.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Modified-DH parameters (per Franka spec), in meters and radians.
# Index i uses a_{i-1}, alpha_{i-1}, d_i. Length 7 for the arm.
PANDA_A: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088)
PANDA_ALPHA: tuple[float, ...] = (
    0.0,
    -torch.pi / 2,
    torch.pi / 2,
    torch.pi / 2,
    -torch.pi / 2,
    torch.pi / 2,
    torch.pi / 2,
)
PANDA_D: tuple[float, ...] = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0)


def _mdh_transform(a: float, alpha: float, d: float, theta: Tensor) -> Tensor:
    """Modified-DH link transform. theta: (...,) -> (..., 4, 4)."""
    ca, sa = torch.cos(theta), torch.sin(theta)
    cosA, sinA = torch.cos(torch.tensor(alpha, dtype=theta.dtype, device=theta.device)), torch.sin(
        torch.tensor(alpha, dtype=theta.dtype, device=theta.device)
    )
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)

    # Craig modified-DH closed form:
    # [ cos(th),          -sin(th),         0,        a            ]
    # [ sin(th)*cos(al),   cos(th)*cos(al), -sin(al), -sin(al)*d   ]
    # [ sin(th)*sin(al),   cos(th)*sin(al),  cos(al),  cos(al)*d   ]
    # [ 0,                 0,                0,        1           ]
    a_t = a * one
    d_t = d * one
    cA = cosA * one
    sA = sinA * one

    row0 = torch.stack([ca, -sa, zero, a_t], dim=-1)
    row1 = torch.stack([sa * cA, ca * cA, -sA, -sA * d_t], dim=-1)
    row2 = torch.stack([sa * sA, ca * sA, cA, cA * d_t], dim=-1)
    row3 = torch.stack([zero, zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def fk_ee_pose(q: Tensor, flange_offset: float = 0.107) -> Tensor:
    """q: (..., 7) joint angles [rad]. Returns (..., 4, 4) base_from_flange."""
    if q.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {tuple(q.shape)}")

    batch_shape = q.shape[:-1]
    T = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).contiguous()

    for i in range(7):
        Ti = _mdh_transform(PANDA_A[i], PANDA_ALPHA[i], PANDA_D[i], q[..., i])
        T = T @ Ti

    # Fixed flange offset along z of link 7 frame (panda_link7 -> panda_link8).
    flange = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).contiguous()
    # Build a broadcastable [0,0,flange_offset,1] translation.
    flange = flange.clone()
    flange[..., 2, 3] = flange_offset
    return T @ flange


def fk_ee_position(q: Tensor, flange_offset: float = 0.107) -> Tensor:
    """q: (..., 7) joint angles [rad]. Returns (..., 3) EE position in base frame."""
    return fk_ee_pose(q, flange_offset=flange_offset)[..., :3, 3]
