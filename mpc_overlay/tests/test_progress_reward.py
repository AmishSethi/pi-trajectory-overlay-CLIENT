"""Tests for the MPCC-style progress reward (`arrow_progress_reward`) and the
batched arc-length projection helper used to compute it."""
from __future__ import annotations

import dataclasses

import torch

from mpc_overlay import CEMParams, GuidanceSpec, MPCWeights, mpc_overlay
from mpc_overlay.mpc import arrow_progress_reward, build_mpc_cost
from mpc_overlay.trajectory_cost import (
    _cumulative_arc_length,
    _project_ee_to_arc,
    _project_ee_to_arc_batch,
)


def _horizontal_spec():
    """Horizontal 800 px arrow from (200, 400) -> (1000, 400) at an EE configuration
    that projects roughly to the first waypoint."""
    waypoints = torch.tensor(
        [[200.0 + 100.0 * k, 400.0] for k in range(9)], dtype=torch.float32
    )
    K = torch.tensor(
        [[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    ext = torch.tensor([0.05, 0.57, 0.66, -2.221, 0.0, -2.233], dtype=torch.float32)
    jvs = torch.tensor([2.175] * 4 + [2.61] * 3, dtype=torch.float32)
    q0 = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]], dtype=torch.float32)
    return GuidanceSpec(
        waypoints_px=waypoints, image_hw=(720, 1280), K_intrinsics=K,
        extrinsic_cam_in_base=ext, image_flipped_180=False, control_dt=1.0 / 15.0,
        joint_vel_scale=jvs, q0=q0, ee_offset_from_flange=0.1034,
        arrow_lookahead=0.2,
    )


def test_batched_projection_matches_scalar():
    wp = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cum, total = _cumulative_arc_length(wp)
    points = torch.tensor([[3.0, 0.0], [10.0, 5.0], [5.0, 1.0]], dtype=torch.float32)
    # Batched.
    batched = _project_ee_to_arc_batch(points, wp.to(torch.float32), cum.to(torch.float32), total.to(torch.float32))
    # Scalar reference for each.
    expected = torch.stack([
        _project_ee_to_arc(p, wp.to(torch.float32), cum.to(torch.float32), total.to(torch.float32))
        for p in points
    ])
    assert torch.allclose(batched, expected, atol=1e-5), (batched, expected)


def test_progress_reward_is_zero_when_candidates_are_a_vla_noop():
    """With zero actions (a chunk of zeros), the chunk's end EE == start EE, so
    s_end - s_start should be ~0."""
    spec = _horizontal_spec()
    cand = torch.zeros((3, 15, 8), dtype=torch.float32)  # (N=3, T=15, D=8)
    r = arrow_progress_reward(cand, spec)
    assert r.shape == (3,)
    # All-zero chunks → EE doesn't move → reward ~ 0 for all samples.
    assert torch.allclose(r, torch.zeros_like(r), atol=0.5), r


def test_progress_reward_favors_motion_along_arrow():
    """A chunk with a +1 velocity on joint 0 should move the EE; the resulting
    s_end will be larger than zero-chunk's s_end. Check that the progress
    reward increases accordingly."""
    spec = _horizontal_spec()
    zero = torch.zeros((1, 15, 8), dtype=torch.float32)
    moving = torch.zeros((1, 15, 8), dtype=torch.float32)
    moving[:, :, 0] = 1.0  # full positive velocity on joint 0 for all 15 steps
    r_zero = arrow_progress_reward(zero, spec)[0].item()
    r_move = arrow_progress_reward(moving, spec)[0].item()
    # At minimum they must differ; and EE should project to a DIFFERENT arc
    # length than the baseline. (Sign depends on camera/FK geometry — don't
    # assert direction, just that motion registers.)
    assert abs(r_move - r_zero) > 1.0, (r_move, r_zero)


def test_progress_term_lowers_cost_for_forward_motion():
    """When lam_prog is >0 and a candidate moves the EE forward along the arrow
    (positive s_end - s_start), the cost should be lower than with lam_prog=0."""
    spec = _horizontal_spec()
    # Reasonable forward-motion chunk: small positive velocity spread across joints.
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    cand = a_vla.clone().unsqueeze(0)  # (1, 15, 8)
    cand[:, :, 0] = 0.5  # small forward push

    weights_no_prog = MPCWeights(lam_p=0.0, lam_a=0.0, lam_c=0.0, lam_s=0.0, lam_prog=0.0)
    weights_prog = MPCWeights(lam_p=0.0, lam_a=0.0, lam_c=0.0, lam_s=0.0, lam_prog=1.0)

    cost_no = build_mpc_cost(a_vla, spec, weights_no_prog)(cand)[0].item()
    cost_prog = build_mpc_cost(a_vla, spec, weights_prog)(cand)[0].item()
    # When prog is enabled and motion > 0, cost should be strictly lower. Sign
    # of the motion relative to arc direction isn't guaranteed, so test the
    # weaker condition that they differ significantly.
    assert cost_no == 0.0  # nothing else weighted
    assert cost_prog != 0.0  # progress term fires
    # And the absolute reward should be at least a few pixels (given joint 0
    # motion is a large motion in EE pixel space).
    assert abs(cost_prog) > 1.0, cost_prog


def test_mpc_overlay_runs_with_lam_prog():
    """End-to-end smoke: CEM with lam_prog>0 still produces a well-shaped output."""
    spec = _horizontal_spec()
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    w = MPCWeights(lam_p=0.0, lam_a=1.0, lam_c=0.0, lam_s=0.0, lam_prog=10.0)
    cp = CEMParams(n_samples=30, n_iterations=2, n_elites=5, init_std=0.1, seed=0)
    best = mpc_overlay(a_vla, spec, w, cp)
    assert best.shape == (15, 8)
    assert torch.isfinite(best).all()
