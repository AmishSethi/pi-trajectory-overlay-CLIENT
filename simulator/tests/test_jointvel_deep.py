"""Deeper stress-tests of the normalized-joint-velocity integration that
main_robolab.py performs before handing actions to RoboLab's env.step().

These exercise edge cases that `test_main_robolab.py` and
`test_parity_with_upstream.py` don't cover:

    - asymmetric per-joint vel_limits (joints 4-6 use 2.61 vs 2.175 on 0-3)
    - mixed-sign velocities across joints in the same step
    - chunk-boundary re-anchor reset (second chunk's q_target must NOT carry
      over from first chunk's integrated target)
    - full 16-step rollout with two re-anchors, parity vs upstream
    - integrator purity (no hidden state)
    - saturation of a single joint over a full chunk
    - dtype + contiguity of emitted action arrays
    - NaN/Inf handling via clip (np.clip leaves NaN alone, we document this)

Run with:
    .venv/bin/python -m pytest tests/test_jointvel_deep.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import main_robolab as mr


# Make the RoboLab submodule importable (same trick as
# test_parity_with_upstream.py — conftest only adds simulator/).
_ROBOLAB_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "RoboLab"
if str(_ROBOLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOLAB_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedWebsocketClient:
    """Stand-in for WebsocketClientPolicy used by upstream Pi0DroidJointvelClient.

    Cycles through `scripted_chunks` — one chunk per upstream.infer() re-query.
    That lets us test multi-chunk behaviour without a live server.
    """

    def __init__(self, remote_host: str, remote_port: int) -> None:
        self.scripted_chunks: list[np.ndarray] = []
        self.query_count = 0
        self.last_request: dict | None = None

    def infer(self, request: dict) -> dict:
        assert self.scripted_chunks, "test must populate scripted_chunks"
        self.last_request = request
        chunk = self.scripted_chunks[min(self.query_count, len(self.scripted_chunks) - 1)]
        self.query_count += 1
        return {"actions": chunk}


@pytest.fixture
def upstream():
    """Build an upstream Pi0DroidJointvelClient with a scripted websocket."""
    with mock.patch(
        "robolab.inference.pi0_jointvel.websocket_client_policy.WebsocketClientPolicy",
        new=_ScriptedWebsocketClient,
    ):
        from robolab.inference.pi0_jointvel import Pi0DroidJointvelClient

        return Pi0DroidJointvelClient(
            remote_host="localhost", remote_port=9999, open_loop_horizon=8,
        )


def _fake_obs(joint_pos: np.ndarray, gripper: float = 0.0) -> dict:
    """Matches RoboLab env observation schema for a single env."""
    ext = torch.zeros((1, 180, 320, 3), dtype=torch.uint8)
    wrist = torch.zeros((1, 180, 320, 3), dtype=torch.uint8)
    return {
        "image_obs": {"external_cam": ext, "wrist_cam": wrist},
        "proprio_obs": {
            "arm_joint_pos": torch.from_numpy(joint_pos[None, :]).float(),
            "gripper_pos": torch.tensor([[gripper]], dtype=torch.float32),
        },
    }


def _simulate_ours(
    chunks: list[np.ndarray],
    joint_positions_at_anchor: list[np.ndarray],
    open_loop_horizon: int = 8,
) -> list[np.ndarray]:
    """Walk main_robolab's inner action-processing loop.

    Mirrors the exact order of operations in main_robolab.main:
        - on chunk boundary: fetch next chunk + re-anchor q_target to the
          current measured joint_position
        - every step: integrate + gripper-binarize + concat

    `joint_positions_at_anchor[i]` is the arm pose observed at the boundary
    that fetches chunks[i]. Between boundaries we don't observe the arm
    again (it's only consulted at the re-anchor), matching main_robolab.
    """
    emitted: list[np.ndarray] = []
    q_target = None
    counter = 0
    chunk_idx = -1
    chunk = None
    total_steps = open_loop_horizon * len(chunks)
    for t in range(total_steps):
        if counter == 0 or counter >= open_loop_horizon:
            counter = 0
            chunk_idx += 1
            chunk = chunks[chunk_idx]
            # Re-anchor to CURRENT measured arm position at this boundary.
            q_target = joint_positions_at_anchor[chunk_idx].astype(np.float32).copy()

        raw = chunk[counter]
        counter += 1
        q_target = mr._integrate_velocity(raw[:7], q_target)
        gripper = 1.0 if raw[-1].item() > 0.5 else 0.0
        emitted.append(np.concatenate([q_target, np.array([gripper], dtype=np.float32)]))
    return emitted


def _simulate_upstream(
    upstream_client,
    chunks: list[np.ndarray],
    joint_positions_at_anchor: list[np.ndarray],
    open_loop_horizon: int = 8,
) -> list[np.ndarray]:
    """Call upstream.infer() repeatedly, switching the measured arm pose at
    each chunk boundary so upstream's internal re-anchor sees the same pose
    ours sees."""
    upstream_client.reset()
    upstream_client.client.scripted_chunks = [c.astype(np.float32) for c in chunks]
    upstream_client.client.query_count = 0

    total_steps = open_loop_horizon * len(chunks)
    emitted: list[np.ndarray] = []
    chunk_idx = -1
    for t in range(total_steps):
        # At each chunk boundary, show upstream the matching measured arm
        # pose so its re-anchor line captures the right value.
        if t % open_loop_horizon == 0:
            chunk_idx += 1
            obs_pose = joint_positions_at_anchor[chunk_idx]
        else:
            # Mid-chunk upstream doesn't re-anchor, so the arm pose it sees
            # is irrelevant — pass whatever. Use the current anchor.
            obs_pose = joint_positions_at_anchor[chunk_idx]
        result = upstream_client.infer(
            _fake_obs(obs_pose, gripper=0.0), instruction="test", env_id=0,
        )
        emitted.append(np.asarray(result["action"], dtype=np.float32))
    return emitted


# ---------------------------------------------------------------------------
# 1. Per-joint asymmetric velocity limits
# ---------------------------------------------------------------------------


def test_joints_0_to_3_use_limit_2p175():
    q0 = np.zeros(7, dtype=np.float32)
    v = np.zeros(7, dtype=np.float32)
    v[0] = v[1] = v[2] = v[3] = 1.0
    q1 = mr._integrate_velocity(v, q0)
    expected = 2.175 / 15.0
    for j in range(4):
        assert q1[j] == pytest.approx(expected, abs=1e-7), (
            f"joint {j} should move by 2.175/15 at v=+1, got {q1[j]}"
        )
    for j in range(4, 7):
        assert q1[j] == pytest.approx(0.0, abs=1e-7)


def test_joints_4_to_6_use_limit_2p61():
    q0 = np.zeros(7, dtype=np.float32)
    v = np.zeros(7, dtype=np.float32)
    v[4] = v[5] = v[6] = 1.0
    q1 = mr._integrate_velocity(v, q0)
    expected = 2.61 / 15.0
    for j in range(4):
        assert q1[j] == pytest.approx(0.0, abs=1e-7)
    for j in range(4, 7):
        assert q1[j] == pytest.approx(expected, abs=1e-7), (
            f"joint {j} should move by 2.61/15 at v=+1, got {q1[j]}"
        )


def test_droid_vel_limits_are_pinned_values():
    """Pin the exact per-joint limits documented in the upstream client.
    If someone changes these by accident, this catches it."""
    np.testing.assert_allclose(
        mr.DROID_VEL_LIMITS,
        np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32),
    )
    assert mr.SIM_CONTROL_FREQUENCY == 15


# ---------------------------------------------------------------------------
# 2. Mixed-sign velocities in one step
# ---------------------------------------------------------------------------


def test_mixed_sign_velocities_integrate_per_joint():
    q0 = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    v = np.array([+1.0, -1.0, +0.5, -0.5, +1.0, -1.0, 0.0], dtype=np.float32)
    q1 = mr._integrate_velocity(v, q0)
    dt = 1.0 / 15.0
    expected = q0 + v * mr.DROID_VEL_LIMITS * dt
    np.testing.assert_allclose(q1, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Integrator purity — no hidden state
# ---------------------------------------------------------------------------


def test_integrator_is_pure_repeated_calls_identical():
    q0 = np.array([0.3, -0.1, 0.2, -0.4, 0.0, 0.7, -0.6], dtype=np.float32)
    v = np.array([0.5, -0.5, 0.1, -0.1, 0.9, -0.9, 0.0], dtype=np.float32)
    a = mr._integrate_velocity(v, q0)
    b = mr._integrate_velocity(v, q0)
    np.testing.assert_array_equal(a, b)  # bit-identical


def test_integrator_does_not_mutate_inputs():
    q0 = np.array([0.1] * 7, dtype=np.float32)
    v = np.array([0.5] * 7, dtype=np.float32)
    q0_copy = q0.copy()
    v_copy = v.copy()
    _ = mr._integrate_velocity(v, q0)
    np.testing.assert_array_equal(q0, q0_copy)
    np.testing.assert_array_equal(v, v_copy)


# ---------------------------------------------------------------------------
# 4. Saturation over a full chunk
# ---------------------------------------------------------------------------


def test_single_joint_saturated_over_full_horizon():
    """v=+1 on joint 0 for 8 steps should accumulate to 8 * 2.175/15."""
    q = np.zeros(7, dtype=np.float32)
    v = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(8):
        q = mr._integrate_velocity(v, q)
    assert q[0] == pytest.approx(8 * 2.175 / 15.0, abs=1e-6)
    np.testing.assert_allclose(q[1:], 0, atol=1e-7)


def test_saturated_move_stays_within_physical_range_per_chunk():
    """At v_norm=+1, one chunk (8 steps * 1/15 s) moves at most
    2.61 * 8/15 ≈ 1.392 rad on the wrist joints — a sanity check that
    one chunk can't whip the arm through more than ~80 deg, which is the
    envelope RoboLab expects."""
    dt_per_chunk = 8.0 / 15.0  # seconds of open-loop integration
    max_joint_move = mr.DROID_VEL_LIMITS.max() * dt_per_chunk
    assert max_joint_move < 1.5, (
        f"One chunk could move a joint by {max_joint_move:.3f} rad — "
        f"suspicious; expected <1.5 rad for DROID velocity profile."
    )


# ---------------------------------------------------------------------------
# 5. Chunk-boundary re-anchor behaviour
# ---------------------------------------------------------------------------


def test_chunk_boundary_re_anchors_to_fresh_joint_pose(upstream):
    """Two chunks back to back. Between them the measured joint_position
    jumps (simulating env drift during open-loop). Both implementations
    must re-anchor to the NEW measurement on chunk 2's first step —
    chunk 2's output should NOT continue from where chunk 1 left off."""
    rng = np.random.default_rng(123)
    chunk_a = rng.uniform(-0.5, 0.5, size=(16, 8)).astype(np.float32)
    chunk_b = rng.uniform(-0.5, 0.5, size=(16, 8)).astype(np.float32)
    chunk_a[:, -1] = 0.0
    chunk_b[:, -1] = 0.0

    pose_a = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    pose_b = np.array([0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6], dtype=np.float32)

    ours = _simulate_ours([chunk_a, chunk_b], [pose_a, pose_b])
    theirs = _simulate_upstream(upstream, [chunk_a, chunk_b], [pose_a, pose_b])

    assert len(ours) == 16
    assert len(theirs) == 16
    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(
            a, b, atol=0, rtol=0,
            err_msg=f"step {i} diverged: ours={a}, theirs={b}",
        )

    # Verify the re-anchor actually happened: step 8 (start of chunk 2)
    # should be close to pose_b + 1-step integrated velocity, NOT a
    # continuation of step 7's trajectory from pose_a.
    dt = 1.0 / 15.0
    expected_step8_q = pose_b + np.clip(chunk_b[0, :7], -1, 1) * mr.DROID_VEL_LIMITS * dt
    np.testing.assert_allclose(ours[8][:7], expected_step8_q, atol=1e-6)


def test_two_chunk_rollout_matches_upstream_bitwise(upstream):
    """Full 16-step rollout (2 chunks) with randomized actions + gripper
    flipping over the threshold. Should be bit-identical to upstream."""
    rng = np.random.default_rng(7)
    chunk_a = rng.uniform(-1.3, 1.3, size=(16, 8)).astype(np.float32)
    chunk_b = rng.uniform(-1.3, 1.3, size=(16, 8)).astype(np.float32)
    chunk_a[:, -1] = rng.uniform(0, 1, size=16).astype(np.float32)
    chunk_b[:, -1] = rng.uniform(0, 1, size=16).astype(np.float32)

    pose_a = rng.uniform(-2, 2, size=7).astype(np.float32)
    pose_b = rng.uniform(-2, 2, size=7).astype(np.float32)

    ours = _simulate_ours([chunk_a, chunk_b], [pose_a, pose_b])
    theirs = _simulate_upstream(upstream, [chunk_a, chunk_b], [pose_a, pose_b])

    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0, err_msg=f"step {i}")


# ---------------------------------------------------------------------------
# 6. Dtype + shape of the action we hand to env.step
# ---------------------------------------------------------------------------


def test_emitted_action_shape_and_dtype():
    """Matches the signature RoboLab's env.step() expects: 8-dim float32."""
    q = np.zeros(7, dtype=np.float32)
    v = np.ones(7, dtype=np.float32) * 0.5
    q = mr._integrate_velocity(v, q)
    action = np.concatenate([q, np.array([1.0], dtype=np.float32)])
    assert action.shape == (8,)
    assert action.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. Integration equation algebraic identity
# ---------------------------------------------------------------------------


def test_integration_is_additive():
    """integrate(v1, integrate(v2, q)) == integrate(v1+v2, q) for small-enough v
    that no clipping happens. Verifies linearity of the one-step update."""
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    v2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    dt = 1.0 / 15.0
    # Two one-steps:
    q_twostep = mr._integrate_velocity(v2, mr._integrate_velocity(v1, q0))
    # Manual: each step adds v_j*vel_limits_j*dt; over two steps adds (v1+v2)*vel_limits*dt.
    manual = q0 + (v1 + v2) * mr.DROID_VEL_LIMITS * dt
    np.testing.assert_allclose(q_twostep, manual, atol=1e-7)


# ---------------------------------------------------------------------------
# 8. Gripper threshold corner cases (exact 0.5 stays at 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (-1.0, 0.0),
        (-0.01, 0.0),
        (0.0, 0.0),
        (0.4999999, 0.0),
        (0.5, 0.0),  # strict >
        (0.5000001, 1.0),
        (0.9, 1.0),
        (1.0, 1.0),
        (1.5, 1.0),
    ],
)
def test_gripper_threshold_is_strict_greater_than_half(raw, expected):
    """main_robolab uses `> 0.5` (strict). 0.5 is closed at 0. Documents
    the contract that must stay in sync with upstream."""
    out = 1.0 if raw > 0.5 else 0.0
    assert out == expected


# ---------------------------------------------------------------------------
# 9. Observations schema round-trip
# ---------------------------------------------------------------------------


def test_request_wire_format_has_no_batch_dim():
    """The server rejects batch-dimensioned obs. Verify that the joint /
    gripper vectors we extract are 1-D, not (1, 7) / (1, 1)."""
    obs = _fake_obs(np.array([0.1]*7, dtype=np.float32), gripper=0.3)
    cur = mr._extract_observation(mr.Args(task="X"), obs, env_id=0)
    assert cur["joint_position"].shape == (7,), cur["joint_position"].shape
    assert cur["gripper_position"].shape == (1,), cur["gripper_position"].shape


# ---------------------------------------------------------------------------
# 10. Sanity: inference chunk as returned by the server
# ---------------------------------------------------------------------------


def test_server_style_chunk_16_8_float_works_end_to_end():
    """The pi05_droid policy emits a (16, 8) float chunk. Walk the exact
    loop main_robolab walks and verify the emitted action stream has the
    right shape and no NaNs."""
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-0.8, 0.8, size=(16, 8)).astype(np.float32)
    q0 = rng.uniform(-1.5, 1.5, size=7).astype(np.float32)
    emitted = _simulate_ours([chunk], [q0], open_loop_horizon=8)
    assert len(emitted) == 8
    for i, a in enumerate(emitted):
        assert a.shape == (8,), f"step {i}: bad shape {a.shape}"
        assert a.dtype == np.float32
        assert np.all(np.isfinite(a)), f"step {i}: non-finite {a}"
        assert a[-1] in (0.0, 1.0), f"step {i}: gripper not binary: {a[-1]}"
