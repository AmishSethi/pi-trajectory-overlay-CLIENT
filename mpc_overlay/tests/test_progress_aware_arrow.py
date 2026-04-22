"""Tests for the sliding-window (progress-aware) arrow target.

Covers:
  (1) Legacy behaviour (arrow_lookahead=None) matches the original resampler.
  (2) With lookahead=1.0, target spans [s0, total] — i.e. when s0=0 it equals legacy.
  (3) When EE is at arrow midpoint, target starts at midpoint.
  (4) With small lookahead, all target points cluster near s0.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from mpc_overlay import GuidanceSpec, predict_ee_pixels
from mpc_overlay.trajectory_cost import (
    _cumulative_arc_length,
    _project_ee_to_arc,
    _resample_waypoints,
    _sample_arrow_segment,
    build_arrow_target,
    ee_pixel_at_q0,
    trajectory_cost,
)


def _make_spec(arrow_lookahead=None, ee_offset=0.1034, waypoints=None,
               q0=None):
    # A simple arrow across the image: horizontal line from (200, 400) to (1000, 400)
    if waypoints is None:
        waypoints = torch.tensor(
            [[200.0 + 100.0 * k, 400.0] for k in range(9)], dtype=torch.float32
        )
    K = torch.tensor(
        [[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    ext = torch.tensor([0.05, 0.57, 0.66, -2.221, 0.0, -2.233], dtype=torch.float32)
    jvs = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=torch.float32)
    if q0 is None:
        q0 = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]], dtype=torch.float32)
    return GuidanceSpec(
        waypoints_px=waypoints,
        image_hw=(720, 1280),
        K_intrinsics=K,
        extrinsic_cam_in_base=ext,
        image_flipped_180=False,
        control_dt=1.0 / 15.0,
        joint_vel_scale=jvs,
        q0=q0,
        waypoint_mask=None,
        ee_offset_from_flange=ee_offset,
        arrow_lookahead=arrow_lookahead,
    )


def test_legacy_behaviour_unchanged():
    spec = _make_spec(arrow_lookahead=None)
    T = 15
    legacy = build_arrow_target(spec, T)
    reference = _resample_waypoints(spec.waypoints_px, T)
    assert torch.allclose(legacy, reference, atol=1e-5)


def test_cumulative_arc_length_is_correct():
    wp = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    cum, total = _cumulative_arc_length(wp)
    assert torch.allclose(cum, torch.tensor([0.0, 3.0, 7.0]))
    assert float(total) == pytest.approx(7.0)


def test_projection_to_arrow_middle():
    wp = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    cum, total = _cumulative_arc_length(wp)
    # Point at (4, 2) — perpendicular foot at (4, 0), arc length 4.
    ee = torch.tensor([4.0, 2.0])
    s0 = _project_ee_to_arc(ee, wp, cum, total)
    assert float(s0) == pytest.approx(4.0, abs=1e-5)


def test_sliding_window_starts_at_projection():
    spec = _make_spec(arrow_lookahead=0.5)
    T = 15
    # Synthesise a current EE pixel that projects to the arrow MIDPOINT.
    # Arrow goes (200, 400) -> (1000, 400) horizontally at y=400, length 800.
    # Midpoint is (600, 400). Arc length at midpoint = 400.
    ee_mid = torch.tensor([600.0, 400.0])
    target = build_arrow_target(spec, T, ee_px_now=ee_mid)
    # target[0] should be ~(600, 400) (the projection point itself).
    assert torch.allclose(target[0], ee_mid, atol=2.0), f"first target = {target[0]}"
    # target[T-1] should be halfway along the remainder of the arrow.
    # s0=400, s1=400+0.5*800=800, so position at arc length 800 = (1000, 400).
    expected_last = torch.tensor([1000.0, 400.0])
    assert torch.allclose(target[-1], expected_last, atol=2.0), f"last target = {target[-1]}"


def test_small_lookahead_clusters_targets_near_s0():
    spec = _make_spec(arrow_lookahead=0.001)
    T = 15
    ee = torch.tensor([500.0, 400.0])
    target = build_arrow_target(spec, T, ee_px_now=ee)
    # All points should be within ~1 px of the projection point (500, 400).
    assert (torch.linalg.vector_norm(target - ee, dim=-1) < 2.0).all()


def test_full_lookahead_matches_legacy_at_s0_zero():
    # When ee projects to s0=0 (arrow start) and lookahead=1.0, target spans
    # the whole arrow, which should match _resample_waypoints.
    spec = _make_spec(arrow_lookahead=1.0)
    T = 15
    ee = spec.waypoints_px[0]  # arrow start
    target = build_arrow_target(spec, T, ee_px_now=ee)
    legacy = _resample_waypoints(spec.waypoints_px, T)
    assert torch.allclose(target, legacy, atol=1e-4)


def test_ee_pixel_at_q0_roundtrip():
    spec = _make_spec(arrow_lookahead=0.2)
    ee_pix = ee_pixel_at_q0(spec)
    assert ee_pix.shape == (2,)
    assert torch.isfinite(ee_pix).all()


def test_trajectory_cost_works_with_sliding_window():
    spec = _make_spec(arrow_lookahead=0.2)
    a = torch.zeros((1, 15, 8), dtype=torch.float32)
    cost = trajectory_cost(a, spec)
    assert cost.ndim == 0
    assert torch.isfinite(cost)


def test_gripper_force_override_near_end_opens():
    """When EE projects near the arrow end, post-CEM override should force open."""
    from mpc_overlay import mpc_overlay
    from mpc_overlay.mpc import MPCWeights, CEMParams
    spec = _make_spec(arrow_lookahead=0.2)
    spec = dataclasses.replace(
        spec, gripper_force_override=True, gripper_zone_frac=0.3
    )
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    a_vla[:, 7] = 1.0  # VLA says CLOSE but EE is at end → override to OPEN
    w = MPCWeights(lam_p=1.0, lam_a=1.0, lam_c=100.0, lam_s=0.01)
    cp = CEMParams(n_samples=30, n_iterations=2, n_elites=5, init_std=0.02, seed=0)
    best = mpc_overlay(a_vla, spec, w, cp)
    # q0 in _make_spec puts EE far to the right, so projection is near arrow end (frac>0.7)
    # → expect all -1 (open)
    assert torch.allclose(best[:, 7], -torch.ones(15), atol=1e-5), best[:, 7]


def test_gripper_force_override_near_start_closes():
    """Flipping the arrow puts EE's projection near the arrow start → force close."""
    from mpc_overlay import mpc_overlay
    from mpc_overlay.mpc import MPCWeights, CEMParams
    spec = _make_spec(arrow_lookahead=0.2)
    flipped_wp = torch.flip(spec.waypoints_px, [0])
    spec = dataclasses.replace(
        spec, waypoints_px=flipped_wp,
        gripper_force_override=True, gripper_zone_frac=0.3
    )
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    a_vla[:, 7] = -1.0  # VLA says OPEN but EE is near (flipped) start → override to CLOSE
    w = MPCWeights(lam_p=1.0, lam_a=1.0, lam_c=100.0, lam_s=0.01)
    cp = CEMParams(n_samples=30, n_iterations=2, n_elites=5, init_std=0.02, seed=0)
    best = mpc_overlay(a_vla, spec, w, cp)
    assert torch.allclose(best[:, 7], torch.ones(15), atol=1e-5), best[:, 7]


def test_gripper_force_override_middle_passthrough():
    """When EE projects into the middle zone (not start/end), gripper is untouched."""
    from mpc_overlay import mpc_overlay
    from mpc_overlay.mpc import MPCWeights, CEMParams
    # Make a VERY long arrow so the EE's 50% projection falls in the middle zone.
    wp = torch.tensor([[100.0 + 50.0 * k, 400.0] for k in range(25)], dtype=torch.float32)
    spec = _make_spec(arrow_lookahead=0.2, waypoints=wp)
    spec = dataclasses.replace(
        spec, gripper_force_override=True, gripper_zone_frac=0.1  # narrow zones
    )
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    a_vla[:, 7] = 0.3  # VLA's chosen value
    w = MPCWeights(lam_p=1.0, lam_a=1.0, lam_c=100.0, lam_s=0.01)
    cp = CEMParams(n_samples=30, n_iterations=2, n_elites=5, init_std=0.02, seed=0)
    best = mpc_overlay(a_vla, spec, w, cp)
    # Middle zone: expect gripper unchanged (0.3). CEM might slightly perturb with
    # freeze_gripper=False, but freeze_gripper defaults to True so gripper == a_vla[:, 7].
    assert torch.allclose(best[:, 7], a_vla[:, 7], atol=1e-5)
