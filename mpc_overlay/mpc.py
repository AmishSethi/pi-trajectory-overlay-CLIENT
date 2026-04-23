"""MPC-overlay optimisation for pi0.5 action chunks.

The VLA stays frozen; we optimise in action space with the VLA's sampled chunk
as a soft prior. Candidate action chunks are scored by a composite cost
(prior + arrow + joint/action constraints + smoothness) and refined with CEM.

Action chunk convention:
  shape (T, D) with D >= 8. First 7 dims are normalized joint velocities in
  [-1, 1]; dim 7 is gripper and is held fixed (never perturbed by CEM).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch
from torch import Tensor

from mpc_overlay.trajectory_cost import GuidanceSpec
from mpc_overlay.trajectory_cost import _resample_waypoints
from mpc_overlay.trajectory_cost import predict_ee_pixels


# Franka Panda joint limits (rad) from the FCI spec.
PANDA_Q_MIN = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_MAX = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclasses.dataclass
class MPCWeights:
    lam_p: float = 1.0
    lam_a: float = 1.0
    lam_c: float = 100.0
    lam_s: float = 0.01
    # Progress reward (MPCC-style). When >0, the cost is decreased by
    # `lam_prog * (s_end - s_start)` where s_* are arc-length projections of
    # the chunk's ending/starting EE pixel onto the arrow. s_end - s_start is
    # a scalar in units of pixels of arrow arc length. 0 = disabled.
    lam_prog: float = 0.0

    # --- Policy-blending arbitration (Dragan & Srinivasa 2013) ------------
    # When ``arbitration_d_grasp_px > 0``, a scalar confidence alpha in
    # [0, 1] is computed per cost call as
    #     alpha = sigmoid((d_crit - arbitration_d_grasp_px) / arbitration_tau_px)
    # where d_crit = min(|ee_now - first_wp|, |ee_now - last_wp|) — the EE's
    # pixel distance to the arrow's pickup or drop endpoint, whichever is
    # closer. alpha→0 "near a critical contact point" (trust the VLA),
    # alpha→1 far from one (trust the MPC arrow).
    #
    # Effective cost inside the cost_fn becomes:
    #     lam_a_eff    = alpha * lam_a
    #     lam_prog_eff = alpha * lam_prog
    #     lam_p_eff    = lam_p + (1 - alpha) * prior_boost_near_waypoint
    #
    # This is the classic "timid-blending" alpha-arbitration used in
    # shared-autonomy literature, adapted for the VLA-MPC setting. When
    # arbitration_d_grasp_px == 0 (default) arbitration is disabled and
    # cost is computed with the raw weights above — identical to pre-
    # arbitration behaviour.
    arbitration_d_grasp_px: float = 0.0
    arbitration_tau_px: float = 15.0
    prior_boost_near_waypoint: float = 0.0


@dataclasses.dataclass
class CEMParams:
    n_samples: int = 200
    n_iterations: int = 4
    n_elites: int = 20
    init_std: float = 0.05
    min_std: float = 1e-3
    freeze_gripper: bool = True
    clip_action: bool = True
    seed: int | None = None

    # --- Trust-region projection (TRPO / residual-MPPI pattern) -----------
    # When > 0, every CEM sample is projected into a ball of L2 radius
    # ``trust_region_radius`` around the VLA prior after adding noise and
    # applying the standard action-box clip. This guarantees no candidate
    # drifts further from a_vla than the semantic radius, independent of
    # how aggressive the arrow / progress weights become. 0 = disabled
    # (legacy behaviour).
    trust_region_radius: float = 0.0


# --------------------------------------------------------------------------- #
# Per-term penalties
# --------------------------------------------------------------------------- #
def prior_penalty(candidates: Tensor, a_vla: Tensor) -> Tensor:
    """L2 deviation of each candidate from the VLA prior, per sample.

    candidates: (N, T, D); a_vla: (T, D). Returns (N,).
    """
    diff = candidates - a_vla.unsqueeze(0)
    return diff.pow(2).sum(dim=(-2, -1))


def smoothness_penalty(candidates: Tensor) -> Tensor:
    """Sum of squared finite differences along the time axis. Returns (N,)."""
    # (N, T-1, D)
    diffs = candidates[:, 1:, :] - candidates[:, :-1, :]
    return diffs.pow(2).sum(dim=(-2, -1))


def action_box_penalty(candidates: Tensor) -> Tensor:
    """ReLU(|a[..., :7]| - 1)^2 summed over T and dims. Returns (N,)."""
    a7 = candidates[..., :7]
    over = torch.relu(a7.abs() - 1.0)
    return over.pow(2).sum(dim=(-2, -1))


def joint_limit_penalty(
    candidates: Tensor,                 # (N, T, D>=7)
    q0: Tensor,                         # (7,) or (1, 7) or (N, 7)
    joint_vel_scale: Tensor,            # (7,)
    control_dt: float,
    q_min: Tensor = PANDA_Q_MIN,
    q_max: Tensor = PANDA_Q_MAX,
) -> Tensor:
    """Integrate joint velocities to a joint trajectory (matching `predict_ee_pixels`)
    and sum squared violations of [q_min, q_max] across all (T, 7) entries.
    Returns (N,).
    """
    device = candidates.device
    dtype = candidates.dtype
    N = candidates.shape[0]

    jvs = joint_vel_scale.to(device=device, dtype=dtype)
    qmin = q_min.to(device=device, dtype=dtype)
    qmax = q_max.to(device=device, dtype=dtype)

    q0_ = q0.to(device=device, dtype=dtype)
    if q0_.dim() == 1:
        q0_ = q0_.unsqueeze(0).expand(N, -1)
    elif q0_.shape[0] == 1 and N != 1:
        q0_ = q0_.expand(N, -1)

    # Same integration as predict_ee_pixels: q_k = q0 + cumsum(a[:, :7] * jvs) * dt
    vel = candidates[..., :7] * jvs                             # (N, T, 7)
    dq = vel * float(control_dt)
    q_traj = q0_.unsqueeze(1) + torch.cumsum(dq, dim=1)         # (N, T, 7)

    low_viol = torch.relu(qmin - q_traj)
    high_viol = torch.relu(q_traj - qmax)
    return (low_viol.pow(2) + high_viol.pow(2)).sum(dim=(-2, -1))


def _resample_mask_to_T(mask: Tensor, spec: GuidanceSpec, T: int) -> Tensor:
    """Resample a per-waypoint mask (K,) to length T along arc length.

    Uses nearest-neighbour against the same arc-length grid as `_resample_waypoints`.
    """
    device = mask.device
    dtype = mask.dtype
    if mask.shape[0] == T:
        return mask
    K = spec.waypoints_px.shape[0]
    if K == 1:
        return mask[0:1].expand(T).clone()
    wp = spec.waypoints_px.detach().to(device=device, dtype=dtype)
    seg = wp[1:] - wp[:-1]
    seg_len = torch.linalg.vector_norm(seg, dim=-1)
    cum = torch.cat([
        torch.zeros(1, dtype=dtype, device=device),
        torch.cumsum(seg_len, dim=0),
    ], dim=0)
    total = cum[-1].clamp(min=1e-12)
    s = cum / total
    t_grid = torch.linspace(0.0, 1.0, T, dtype=dtype, device=device)
    idx = torch.searchsorted(s, t_grid, right=False)
    idx = torch.clamp(idx, min=1, max=K - 1)
    j0 = idx - 1
    j1 = idx
    alpha = ((t_grid - s[j0]) / (s[j1] - s[j0]).clamp(min=1e-12)).clamp(0.0, 1.0)
    nearest = torch.where(alpha < 0.5, j0, j1)
    return mask[nearest]


def arrow_penalty(candidates: Tensor, spec: GuidanceSpec) -> Tensor:
    """Per-sample masked L2 in the flipped pixel frame against a target arrow.

    When ``spec.arrow_lookahead`` is set, the target is a SLIDING WINDOW in arc
    length starting at the current EE projection (progress-aware receding-horizon
    tracking). When None, legacy full-arrow resampling is used.

    candidates: (N, T, D>=8). Returns (N,).
    """
    from mpc_overlay.trajectory_cost import build_arrow_target, ee_pixel_at_q0
    N, T, _ = candidates.shape
    device = candidates.device
    dtype = candidates.dtype

    # Expand q0 from (1, 7) -> (N, 7) via a shallow copy of the spec.
    q0_n = spec.q0.to(device=device, dtype=dtype).expand(N, -1)
    spec_n = dataclasses.replace(spec, q0=q0_n)

    pred = predict_ee_pixels(candidates[..., :8], spec_n)       # (N, T, 2)

    # Current EE pixel from q0 (taking first batch row) — used as the sliding-window
    # anchor. Computed once per cost call (shared across all N candidates).
    spec_current = dataclasses.replace(spec, q0=spec.q0.to(device=device, dtype=dtype)[:1])
    ee_now = ee_pixel_at_q0(spec_current, device=device, dtype=dtype)

    target = build_arrow_target(spec, T, ee_px_now=ee_now, device=device, dtype=dtype)
    err2 = (pred - target.unsqueeze(0)).pow(2).sum(dim=-1)      # (N, T)

    if spec.waypoint_mask is None:
        mask = torch.ones(T, dtype=dtype, device=device)
    else:
        wm = spec.waypoint_mask.to(device=device, dtype=dtype)
        mask = _resample_mask_to_T(wm, spec, T)

    denom = mask.sum().clamp_min(1e-8)
    return (err2 * mask.unsqueeze(0)).sum(dim=-1) / denom


def arrow_progress_reward(candidates: Tensor, spec: GuidanceSpec) -> Tensor:
    """MPCC-style progress reward. Returns (N,) reward in pixels of arc length
    (to be SUBTRACTED from the cost by the caller).

    We use the AVERAGE arc-length gain across the chunk's predicted trajectory
    relative to the starting projection:
        reward = mean_t clamp( s_t - s_0, min=0 )
    This avoids the "fly to a distant endpoint" failure mode you get when you
    only reward s_end: CEM would otherwise pick chunks whose intermediate
    states are wild so long as the endpoint projects far along the arrow.
    clamp(min=0) prevents the reward from going negative on forward-looking
    chunks that temporarily backtrack (regressions get no credit, but aren't
    double-punished).

    Skipped (returns zeros) if waypoints are degenerate (<2 points, zero length).

    candidates: (N, T, D>=7).
    """
    from mpc_overlay.trajectory_cost import (
        _cumulative_arc_length, _project_ee_to_arc,
        _project_ee_to_arc_batch, ee_pixel_at_q0,
    )
    N, T, _ = candidates.shape
    device = candidates.device
    dtype = candidates.dtype
    waypoints = spec.waypoints_px.detach().to(device=device, dtype=dtype)
    if waypoints.shape[0] < 2:
        return torch.zeros(N, dtype=dtype, device=device)
    cum_arc, total = _cumulative_arc_length(waypoints)
    if float(total) <= 0.0:
        return torch.zeros(N, dtype=dtype, device=device)

    # Current (t=0) EE projection — shared anchor across all N candidates.
    spec_current = dataclasses.replace(spec, q0=spec.q0.to(device=device, dtype=dtype)[:1])
    ee_now = ee_pixel_at_q0(spec_current, device=device, dtype=dtype)
    s_start = _project_ee_to_arc(ee_now, waypoints, cum_arc, total)

    q0_n = spec.q0.to(device=device, dtype=dtype).expand(N, -1)
    spec_n = dataclasses.replace(spec, q0=q0_n)
    pred = predict_ee_pixels(candidates[..., :8], spec_n)       # (N, T, 2)
    pred_flat = pred.reshape(N * T, 2)                           # (N*T, 2)
    s_all = _project_ee_to_arc_batch(pred_flat, waypoints, cum_arc, total)  # (N*T,)
    s_all = s_all.reshape(N, T)                                  # (N, T)
    gain = (s_all - s_start).clamp(min=0.0)                      # (N, T); reward forward-only
    return gain.mean(dim=-1)                                     # (N,)


def gripper_action_reward(candidates: Tensor, spec: GuidanceSpec) -> Tensor:
    """Return a scalar reward (to be SUBTRACTED from the cost) that encourages
    gripper closure when the projected EE is near the arrow start, and gripper
    release when near the arrow end. Handles the grasp-stall failure mode we
    observed on BananaInBowl baseline and on high-λ BananaOnPlate runs without
    forcing the VLA's grasp timing.

    candidates: (N, T, D>=8). Returns (N,) non-negative reward — higher = better.
    """
    from mpc_overlay.trajectory_cost import (
        _cumulative_arc_length, _project_ee_to_arc, ee_pixel_at_q0,
    )
    if spec.gripper_reward_weight <= 0.0:
        return torch.zeros(candidates.shape[0],
                           dtype=candidates.dtype, device=candidates.device)

    N, T, D = candidates.shape
    device = candidates.device
    dtype = candidates.dtype
    waypoints = spec.waypoints_px.detach().to(device=device, dtype=dtype)
    cum_arc, total = _cumulative_arc_length(waypoints)
    if float(total) <= 0.0 or waypoints.shape[0] < 2:
        return torch.zeros(N, dtype=dtype, device=device)

    spec_current = dataclasses.replace(spec, q0=spec.q0.to(device=device, dtype=dtype)[:1])
    ee_now = ee_pixel_at_q0(spec_current, device=device, dtype=dtype)
    s0 = _project_ee_to_arc(ee_now, waypoints, cum_arc, total)
    s_frac = float((s0 / total.clamp(min=1e-12)).item())

    zone = float(spec.gripper_zone_frac)
    # Gripper action is channel 7. Binarised at execution by > 0.5 threshold
    # on the real robot; VLA emits continuous values in [-1, 1] so we use sign.
    grip_chunk_mean = candidates[:, :, 7].mean(dim=-1)  # (N,)
    w = float(spec.gripper_reward_weight)

    if s_frac < zone:
        # Near arrow start — reward gripper CLOSE (grip_value > 0.5 desirable).
        return w * torch.clamp(grip_chunk_mean, min=-1.0, max=1.0)
    if s_frac > 1.0 - zone:
        # Near arrow end — reward gripper OPEN (grip_value < 0.5 desirable).
        return w * torch.clamp(-grip_chunk_mean, min=-1.0, max=1.0)
    return torch.zeros(N, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# Composite cost + CEM
# --------------------------------------------------------------------------- #
def compute_arbitration_alpha(spec: GuidanceSpec, weights: MPCWeights,
                               device, dtype) -> float:
    """Dragan-style timid-blending alpha in [0, 1] based on EE pixel distance
    to the closer of the arrow's two endpoints. alpha→0 near a critical
    contact point (arrow pickup or drop), alpha→1 elsewhere.

    Returns 1.0 (no arbitration) when arbitration_d_grasp_px <= 0 or the
    arrow is degenerate. Cheap: one EE projection + two norms per cost call.
    """
    d_crit = float(weights.arbitration_d_grasp_px)
    if d_crit <= 0.0:
        return 1.0
    waypoints = spec.waypoints_px.detach().to(device=device, dtype=dtype)
    if waypoints.shape[0] < 2:
        return 1.0
    from mpc_overlay.trajectory_cost import ee_pixel_at_q0
    spec_current = dataclasses.replace(
        spec, q0=spec.q0.to(device=device, dtype=dtype)[:1]
    )
    ee_now = ee_pixel_at_q0(spec_current, device=device, dtype=dtype)
    d_start = torch.linalg.vector_norm(ee_now - waypoints[0])
    d_end = torch.linalg.vector_norm(ee_now - waypoints[-1])
    d_min = torch.minimum(d_start, d_end)
    tau = max(float(weights.arbitration_tau_px), 1e-3)
    alpha = torch.sigmoid((d_min - d_crit) / tau)
    return float(alpha.item())


def build_mpc_cost(
    a_vla: Tensor,                      # (T, D)
    spec: GuidanceSpec,                 # spec.q0 is (1, 7)
    weights: MPCWeights,
) -> Callable[[Tensor], Tensor]:
    """Factory that returns cost_fn: (N, T, D) -> (N,).

    When ``weights.arbitration_d_grasp_px > 0``, a scalar alpha in [0, 1] is
    computed ONCE per cost call (shared across all N candidates) from the
    current EE's pixel distance to the closer arrow endpoint; alpha then
    scales the arrow and progress terms, and (1 - alpha) adds
    ``prior_boost_near_waypoint`` to the prior term. This is the
    arbitration recipe from the shared-autonomy literature (Dragan 2013,
    VLA-Pilot 2025) adapted to our CEM composite cost.
    """

    def cost_fn(candidates: Tensor) -> Tensor:
        jvs = spec.joint_vel_scale
        q0 = spec.q0
        alpha = compute_arbitration_alpha(spec, weights,
                                           device=candidates.device,
                                           dtype=candidates.dtype)
        lam_p_eff = weights.lam_p + (1.0 - alpha) * float(weights.prior_boost_near_waypoint)
        lam_a_eff = alpha * weights.lam_a
        lam_prog_eff = alpha * weights.lam_prog
        terms = []
        if lam_p_eff != 0.0:
            terms.append(lam_p_eff * prior_penalty(candidates, a_vla))
        if lam_a_eff != 0.0:
            terms.append(lam_a_eff * arrow_penalty(candidates, spec))
        if weights.lam_c != 0.0:
            jl = joint_limit_penalty(candidates, q0, jvs, spec.control_dt)
            ab = action_box_penalty(candidates)
            terms.append(weights.lam_c * (jl + ab))
        if weights.lam_s != 0.0:
            terms.append(weights.lam_s * smoothness_penalty(candidates))
        if lam_prog_eff != 0.0:
            # Reward (sign-flipped: subtract so a bigger s_end - s_start lowers cost).
            terms.append(-lam_prog_eff * arrow_progress_reward(candidates, spec))
        # Gripper reward (sign-flipped: it's a REWARD, so subtract from cost).
        # Only active when spec.gripper_reward_weight > 0 AND freeze_gripper=False in CEM
        # (the weight is a spec field, but the gripper dim only varies if CEM is
        # allowed to sample it — otherwise this is a no-op).
        if getattr(spec, "gripper_reward_weight", 0.0) and getattr(spec, "gripper_reward_weight", 0.0) > 0.0:
            terms.append(-gripper_action_reward(candidates, spec))
        if not terms:
            return torch.zeros(candidates.shape[0], device=candidates.device, dtype=candidates.dtype)
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total

    return cost_fn


def cem_optimize(
    cost_fn: Callable[[Tensor], Tensor],
    init_mean: Tensor,                  # (T, D)
    params: CEMParams,
    device: torch.device | str,
) -> Tensor:
    """CEM over action chunks. Returns the best single candidate seen (T, D)."""
    device = torch.device(device)
    mu = init_mean.clone().to(device)
    D = mu.shape[-1]
    sigma = torch.full_like(mu, params.init_std)
    best_chunk = mu.clone()
    best_cost = float("inf")
    gen = None
    if params.seed is not None:
        gen = torch.Generator(device=device).manual_seed(params.seed)

    # Seed the search with init_mean itself. Without this, CEM can return a
    # chunk worse than a_vla (its init) when lam_a is small and sampling noise
    # dominates -- e.g., sweep 01 mpc_a0 (lam_a=0) still delayed the baseline
    # grasp because the returned chunk was a noisy perturbation of a_vla
    # instead of a_vla itself.
    init_cost = cost_fn(mu.unsqueeze(0))[0]
    if torch.isfinite(init_cost):
        best_cost = init_cost.item()
        best_chunk = mu.detach().clone()

    # Cache a_vla on device for the trust-region projection (reused every iter).
    a_vla_for_tr = init_mean.to(device)
    tr_radius = float(params.trust_region_radius)

    for _ in range(params.n_iterations):
        eps = torch.randn(
            (params.n_samples, *mu.shape),
            device=device,
            dtype=mu.dtype,
            generator=gen,
        )
        C = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)          # (N, T, D)
        if params.freeze_gripper and D > 7:
            C[:, :, 7] = init_mean[:, 7].to(device).unsqueeze(0)
        if tr_radius > 0.0:
            # Project each candidate into the L2 ball of radius tr_radius
            # around a_vla. "delta" is (N, T, D); flattening over (T, D)
            # gives per-sample norms. Samples inside the ball are untouched;
            # samples outside get shrunk back to the boundary.
            delta = C - a_vla_for_tr.unsqueeze(0)
            norm = delta.pow(2).sum(dim=(-2, -1)).sqrt().clamp(min=1e-12)
            scale = (tr_radius / norm).clamp(max=1.0).unsqueeze(-1).unsqueeze(-1)
            C = a_vla_for_tr.unsqueeze(0) + delta * scale
        if params.clip_action:
            C = C.clamp(-1.0, 1.0)

        costs = cost_fn(C)                                      # (N,)

        if torch.isfinite(costs).any():
            best_idx_iter = int(costs.argmin().item())
            if costs[best_idx_iter].item() < best_cost:
                best_cost = costs[best_idx_iter].item()
                best_chunk = C[best_idx_iter].detach().clone()

        elite_idx = costs.argsort()[: params.n_elites]
        elites = C[elite_idx]                                   # (K, T, D)
        mu = elites.mean(dim=0)
        sigma = elites.std(dim=0).clamp_min(params.min_std)

        # Also consider the new elite mean as a candidate. In CEM on smooth
        # costs, mu routinely beats any individual sample once the elite pool
        # concentrates -- without this, "best candidate" is bottlenecked by
        # early wide-sigma draws.
        mu_cand = mu.clone()
        if params.freeze_gripper and D > 7:
            mu_cand[:, 7] = init_mean[:, 7].to(device)
        if tr_radius > 0.0:
            # Same TR projection on the re-evaluation candidate.
            delta = mu_cand - a_vla_for_tr
            norm = delta.pow(2).sum().sqrt().clamp(min=1e-12)
            if norm.item() > tr_radius:
                mu_cand = a_vla_for_tr + delta * (tr_radius / norm)
        if params.clip_action:
            mu_cand = mu_cand.clamp(-1.0, 1.0)
        mu_cost = cost_fn(mu_cand.unsqueeze(0))[0]
        if torch.isfinite(mu_cost) and mu_cost.item() < best_cost:
            best_cost = mu_cost.item()
            best_chunk = mu_cand.detach().clone()

    return best_chunk


def mpc_overlay(
    a_vla: Tensor,                      # (T, D)
    spec: GuidanceSpec,
    weights: MPCWeights,
    cem_params: CEMParams,
) -> Tensor:
    """Public entry. Returns optimised chunk (T, D) on the same device as a_vla."""
    device = a_vla.device

    # Place spec tensors on the same device. We produce a shallow copy to avoid
    # mutating the caller's spec, and in particular leave spec.q0 at its
    # original (1, 7) shape -- arrow_penalty expands it per-call.
    spec_d = dataclasses.replace(
        spec,
        waypoints_px=spec.waypoints_px.to(device=device, dtype=a_vla.dtype),
        K_intrinsics=spec.K_intrinsics.to(device=device, dtype=a_vla.dtype),
        extrinsic_cam_in_base=spec.extrinsic_cam_in_base.to(device=device, dtype=a_vla.dtype),
        joint_vel_scale=spec.joint_vel_scale.to(device=device, dtype=a_vla.dtype),
        q0=spec.q0.to(device=device, dtype=a_vla.dtype),
        waypoint_mask=(
            spec.waypoint_mask.to(device=device, dtype=a_vla.dtype)
            if spec.waypoint_mask is not None
            else None
        ),
    )

    cost_fn = build_mpc_cost(a_vla, spec_d, weights)
    best = cem_optimize(cost_fn, a_vla, cem_params, device=device)

    # Post-CEM hard gripper override (handles the grasp-stall failure mode where
    # the VLA closes the gripper for one chunk and releases it on the next; CEM
    # cannot help since freeze_gripper=True). When the current EE projects within
    # `gripper_zone_frac` of the arrow start, force full close; near the end,
    # force full open.
    if getattr(spec_d, "gripper_force_override", False) and spec_d.waypoints_px.shape[0] >= 2:
        from mpc_overlay.trajectory_cost import (
            _cumulative_arc_length, _project_ee_to_arc, ee_pixel_at_q0,
        )
        cum_arc, total = _cumulative_arc_length(spec_d.waypoints_px.detach())
        if float(total) > 0.0:
            ee_now = ee_pixel_at_q0(spec_d, device=device, dtype=a_vla.dtype)
            s0 = _project_ee_to_arc(ee_now, spec_d.waypoints_px.detach(), cum_arc, total)
            frac = float((s0 / total.clamp(min=1e-12)).item())
            zone = float(spec_d.gripper_zone_frac)
            if best.shape[-1] >= 8:
                if frac < zone:
                    best = best.clone()
                    best[:, 7] = 1.0
                elif frac > 1.0 - zone:
                    best = best.clone()
                    best[:, 7] = -1.0
    return best
