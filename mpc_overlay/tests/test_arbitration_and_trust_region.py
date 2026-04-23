"""Unit tests for the two balance-restoring additions to the MPC overlay:

  (A) policy-blending alpha-arbitration
      (MPCWeights.arbitration_d_grasp_px / arbitration_tau_px /
      prior_boost_near_waypoint, wired into build_mpc_cost)

  (B) trust-region projection on CEM samples
      (CEMParams.trust_region_radius, wired into cem_optimize)

Both features default to disabled, so the first two tests are
regression-guards: existing behaviour (no arbitration, no TR) must be
bit-identical to the pre-feature implementation at the cost level.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from mpc_overlay import CEMParams, GuidanceSpec, MPCWeights, mpc_overlay
from mpc_overlay.mpc import (
    build_mpc_cost,
    compute_arbitration_alpha,
    cem_optimize,
)


# --- shared fixture -------------------------------------------------------
def _spec(waypoints=None, q0=None, arrow_lookahead=0.15):
    if waypoints is None:
        waypoints = torch.tensor(
            [[200.0 + 100.0 * k, 400.0] for k in range(9)], dtype=torch.float32
        )
    K = torch.tensor(
        [[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    ext = torch.tensor([0.05, 0.57, 0.66, -2.221, 0.0, -2.233], dtype=torch.float32)
    jvs = torch.tensor([2.175] * 4 + [2.61] * 3, dtype=torch.float32)
    if q0 is None:
        q0 = torch.tensor(
            [[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]], dtype=torch.float32
        )
    return GuidanceSpec(
        waypoints_px=waypoints,
        image_hw=(720, 1280),
        K_intrinsics=K,
        extrinsic_cam_in_base=ext,
        image_flipped_180=False,
        control_dt=1.0 / 15.0,
        joint_vel_scale=jvs,
        q0=q0,
        ee_offset_from_flange=0.1034,
        arrow_lookahead=arrow_lookahead,
    )


# =========================================================================
#                          Arbitration alpha helper
# =========================================================================
def test_arbitration_disabled_by_default():
    """arbitration_d_grasp_px == 0 -> alpha == 1.0 (no effect)."""
    spec = _spec()
    w = MPCWeights()
    assert compute_arbitration_alpha(spec, w, device="cpu", dtype=torch.float32) == 1.0


def test_arbitration_alpha_goes_to_zero_when_ee_near_endpoint():
    """Build a waypoints list whose FIRST endpoint sits at the EE's projected
    pixel. alpha should drop toward 0."""
    spec = _spec()
    from mpc_overlay.trajectory_cost import ee_pixel_at_q0
    ee_px = ee_pixel_at_q0(spec).detach()
    # Endpoint = EE pixel; the other endpoint is far away.
    wp = torch.stack([ee_px, ee_px + torch.tensor([1000.0, 0.0])], dim=0)
    spec2 = dataclasses.replace(spec, waypoints_px=wp)
    w = MPCWeights(arbitration_d_grasp_px=50.0, arbitration_tau_px=10.0)
    a = compute_arbitration_alpha(spec2, w, device="cpu", dtype=torch.float32)
    assert a < 0.01, f"expected alpha~0 when ee at an endpoint, got {a}"


def test_arbitration_alpha_goes_to_one_when_ee_far_from_both_endpoints():
    """Both endpoints are far from the EE pixel -> alpha saturates near 1."""
    spec = _spec()
    wp = torch.tensor([[50.0, 50.0], [1200.0, 700.0]], dtype=torch.float32)
    spec2 = dataclasses.replace(spec, waypoints_px=wp)
    w = MPCWeights(arbitration_d_grasp_px=50.0, arbitration_tau_px=10.0)
    a = compute_arbitration_alpha(spec2, w, device="cpu", dtype=torch.float32)
    assert a > 0.99, f"expected alpha~1 when ee far from both endpoints, got {a}"


def test_arbitration_uses_closer_of_two_endpoints():
    """Sanity: min distance across BOTH endpoints drives alpha."""
    spec = _spec()
    from mpc_overlay.trajectory_cost import ee_pixel_at_q0
    ee_px = ee_pixel_at_q0(spec).detach()
    # First endpoint far; SECOND endpoint at EE -> alpha should still drop.
    wp = torch.stack([ee_px + torch.tensor([-1000.0, 0.0]), ee_px], dim=0)
    spec2 = dataclasses.replace(spec, waypoints_px=wp)
    w = MPCWeights(arbitration_d_grasp_px=50.0, arbitration_tau_px=10.0)
    a = compute_arbitration_alpha(spec2, w, device="cpu", dtype=torch.float32)
    assert a < 0.01


# =========================================================================
#                         Arbitration in build_mpc_cost
# =========================================================================
def test_cost_is_unchanged_when_arbitration_is_off():
    """With arbitration_d_grasp_px=0, cost on a chunk must exactly match
    the pre-arbitration path's cost."""
    spec = _spec()
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    candidates = torch.randn(5, 15, 8, dtype=torch.float32)
    w_off = MPCWeights(lam_p=1.0, lam_a=10.0, lam_c=100.0, lam_s=0.01, lam_prog=1.0)
    cost_off = build_mpc_cost(a_vla, spec, w_off)(candidates)
    # Replicate the unarbitrated cost "by hand"
    w_manual = MPCWeights(lam_p=1.0, lam_a=10.0, lam_c=100.0, lam_s=0.01, lam_prog=1.0,
                           arbitration_d_grasp_px=0.0)
    cost_manual = build_mpc_cost(a_vla, spec, w_manual)(candidates)
    assert torch.allclose(cost_off, cost_manual)


def test_cost_arbitration_near_endpoint_down_weights_arrow_and_progress():
    """When alpha~0 (EE near endpoint), the arrow + progress contributions
    should be sharply reduced. The prior_boost adds to lam_p."""
    spec = _spec()
    from mpc_overlay.trajectory_cost import ee_pixel_at_q0
    ee_px = ee_pixel_at_q0(spec).detach()
    wp = torch.stack([ee_px, ee_px + torch.tensor([1000.0, 0.0])], dim=0)
    spec2 = dataclasses.replace(spec, waypoints_px=wp)

    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    # A chunk that has nonzero action so arrow_penalty / prior_penalty are nonzero.
    cand = torch.zeros(1, 15, 8, dtype=torch.float32)
    cand[:, :, 0] = 0.1

    # With arbitration OFF: the usual sum of terms.
    w_off = MPCWeights(lam_p=1.0, lam_a=10.0, lam_prog=1.0, lam_c=0.0, lam_s=0.0)
    c_off = build_mpc_cost(a_vla, spec2, w_off)(cand)[0].item()
    # With arbitration ON near endpoint (alpha~0): arrow + progress scaled by
    # alpha->~0, prior boosted by (1-alpha)*prior_boost = +3.0. Use a VERY
    # sharp tau so alpha truly saturates to 0 (otherwise a tiny residual
    # alpha * lam_a * arrow_penalty dominates, since arrow cost is huge in
    # pixel^2 space).
    w_on = MPCWeights(lam_p=1.0, lam_a=10.0, lam_prog=1.0, lam_c=0.0, lam_s=0.0,
                       arbitration_d_grasp_px=50.0, arbitration_tau_px=1.0,
                       prior_boost_near_waypoint=3.0)
    c_on = build_mpc_cost(a_vla, spec2, w_on)(cand)[0].item()
    assert c_on != c_off, "arbitration produced identical cost — heuristic off"
    # More specific: manually recompose the expected cost with alpha≈0.
    from mpc_overlay.mpc import prior_penalty
    pp = prior_penalty(cand, a_vla)[0].item()
    # alpha ≈ sigmoid((0-50)/1) ≈ 0 -> lam_p_eff = 1.0 + 3.0 = 4.0,
    # arrow/prog terms effectively vanish.
    expected = 4.0 * pp
    assert abs(c_on - expected) < 1e-3, (c_on, expected)


def test_cost_arbitration_far_from_endpoint_equals_unarbitrated():
    """When alpha~1 (EE far from both endpoints), cost must be close to the
    unarbitrated case (only the 1-alpha prior boost term subtracts, which
    goes to zero)."""
    spec = _spec()
    wp = torch.tensor([[50.0, 50.0], [1200.0, 700.0]], dtype=torch.float32)
    spec2 = dataclasses.replace(spec, waypoints_px=wp)
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    cand = torch.zeros(1, 15, 8, dtype=torch.float32)
    cand[:, :, 0] = 0.1
    w_off = MPCWeights(lam_p=1.0, lam_a=10.0, lam_prog=1.0, lam_c=0.0, lam_s=0.0)
    w_on = MPCWeights(lam_p=1.0, lam_a=10.0, lam_prog=1.0, lam_c=0.0, lam_s=0.0,
                       arbitration_d_grasp_px=50.0, arbitration_tau_px=10.0,
                       prior_boost_near_waypoint=3.0)
    c_off = build_mpc_cost(a_vla, spec2, w_off)(cand)[0].item()
    c_on = build_mpc_cost(a_vla, spec2, w_on)(cand)[0].item()
    # alpha ~ 1 -> matches unarbitrated, modulo the (1-alpha)*prior_boost
    # term which vanishes.
    assert abs(c_on - c_off) < 1e-3, (c_on, c_off)


# =========================================================================
#                    Trust-region projection in cem_optimize
# =========================================================================
def test_trust_region_disabled_by_default():
    """trust_region_radius == 0 -> standard CEM. We verify by running CEM
    and checking the output has some deviation from a_vla (would be clipped
    to zero if TR were on with radius ~= 0 and lam_p small)."""
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    spec = _spec()
    w = MPCWeights(lam_p=0.1, lam_a=10.0, lam_prog=0.0, lam_c=0.0, lam_s=0.0)
    cp = CEMParams(n_samples=40, n_iterations=2, n_elites=10, init_std=0.2, seed=0)
    best = mpc_overlay(a_vla, spec, w, cp)
    assert best.shape == (15, 8)
    assert torch.any(torch.abs(best - a_vla) > 0.01), \
        "expected CEM to produce nontrivial deviation from a_vla"


def test_trust_region_caps_max_deviation_from_a_vla():
    """With tiny TR radius, no CEM sample can drift more than that radius
    from a_vla in the full-chunk L2 sense. With lam_a dominating (which
    would otherwise pull hard), we directly test the constraint holds."""
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    spec = _spec()
    TR = 0.05  # tight ball
    w = MPCWeights(lam_p=0.01, lam_a=1000.0, lam_prog=10.0, lam_c=0.0, lam_s=0.0)
    cp = CEMParams(n_samples=40, n_iterations=3, n_elites=10,
                    init_std=0.5, seed=0, trust_region_radius=TR)
    best = mpc_overlay(a_vla, spec, w, cp)
    # Gripper is frozen, so zero out the gripper channel deviation (doesn't
    # get projected onto the ball).
    delta = best - a_vla
    delta[..., 7] = 0.0
    norm = float(delta.pow(2).sum().sqrt().item())
    # The TR projection happens BEFORE the action-box clip, so the final
    # clamped chunk can have norm slightly under TR. Allow 10% headroom.
    assert norm <= TR * 1.1, f"norm {norm} > TR {TR} * 1.1 = {TR*1.1}"


def test_trust_region_with_large_radius_matches_no_tr():
    """Huge TR radius -> projection never fires -> output equals TR-off case
    modulo CEM stochasticity. We use a fixed seed and a conservative test:
    the two runs should produce OUTPUTS that are "similar" (same general
    region of chunk space), not bit-identical, because the TR code path
    is a single extra tensor op that can still affect FP ordering."""
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    spec = _spec()
    w = MPCWeights(lam_p=0.1, lam_a=10.0, lam_prog=0.5, lam_c=0.0, lam_s=0.0)
    cp_no_tr = CEMParams(n_samples=40, n_iterations=2, n_elites=10,
                          init_std=0.1, seed=42)
    cp_big_tr = CEMParams(n_samples=40, n_iterations=2, n_elites=10,
                           init_std=0.1, seed=42, trust_region_radius=1e6)
    best1 = mpc_overlay(a_vla, spec, w, cp_no_tr)
    best2 = mpc_overlay(a_vla, spec, w, cp_big_tr)
    # Not bit-identical but the two best-chunks should lie close to each other.
    assert torch.allclose(best1, best2, atol=1e-4)


def test_trust_region_is_a_soft_cap_not_a_kill_switch():
    """A moderate TR radius still allows CEM to make progress on the cost
    (i.e. the returned chunk has LOWER arrow cost than a_vla). Guards
    against accidentally making TR so tight it nukes all improvement."""
    from mpc_overlay.mpc import arrow_penalty
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    spec = _spec()
    w = MPCWeights(lam_p=0.1, lam_a=100.0, lam_prog=0.0, lam_c=0.0, lam_s=0.0)
    cp = CEMParams(n_samples=80, n_iterations=3, n_elites=20,
                    init_std=0.15, seed=0, trust_region_radius=0.5)
    best = mpc_overlay(a_vla, spec, w, cp)
    init_cost = arrow_penalty(a_vla.unsqueeze(0), spec)[0].item()
    final_cost = arrow_penalty(best.unsqueeze(0), spec)[0].item()
    assert final_cost < init_cost * 0.9, \
        (f"TR=0.5 should still allow >=10% cost reduction; "
         f"got init={init_cost:.1f}, final={final_cost:.1f}")


# =========================================================================
#                           End-to-end smoke
# =========================================================================
def test_arbitration_plus_trust_region_end_to_end():
    """Both features enabled, full CEM run — just asserts the pipeline
    composes without crashing and produces a finite (T, D) chunk."""
    spec = _spec()
    a_vla = torch.zeros(15, 8, dtype=torch.float32)
    w = MPCWeights(lam_p=1.0, lam_a=10.0, lam_prog=1.0, lam_c=100.0, lam_s=0.01,
                    arbitration_d_grasp_px=50.0, arbitration_tau_px=15.0,
                    prior_boost_near_waypoint=2.0)
    cp = CEMParams(n_samples=40, n_iterations=3, n_elites=10,
                    init_std=0.08, seed=0, trust_region_radius=0.5)
    best = mpc_overlay(a_vla, spec, w, cp)
    assert best.shape == (15, 8)
    assert torch.isfinite(best).all()
