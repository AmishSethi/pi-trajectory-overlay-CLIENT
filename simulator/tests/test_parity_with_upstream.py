"""Math parity: our inline _integrate_velocity (main_robolab.py) must be
byte-for-byte equivalent to RoboLab's upstream Pi0DroidJointvelClient from
commit 684d3f2 (`../third_party/RoboLab/robolab/inference/pi0_jointvel.py`).

This pins our math to the exact behaviour the RoboLab team ships with
`--policy pi05_jointvel`, so we can be confident our single-GPU workflow
drives the env the same way their reference eval does.

The upstream client bundles `WebsocketClientPolicy` construction + request
building + velocity integration + gripper binarize into a single `infer()`
call. We side-step the websocket (monkey-patch it to return a scripted
action chunk) so this test runs without a policy server.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import main_robolab as mr


# Make the RoboLab submodule importable in this test process.
_ROBOLAB_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "RoboLab"
if str(_ROBOLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOLAB_ROOT))


class _ScriptedWebsocketClient:
    """Stand-in for openpi_client.websocket_client_policy.WebsocketClientPolicy.

    Returns whatever action chunk the test sets on it. Lets us drive
    Pi0DroidJointvelClient without a running policy server.
    """

    def __init__(self, remote_host: str, remote_port: int) -> None:
        self.scripted_chunk: np.ndarray | None = None
        self.last_request: dict | None = None

    def infer(self, request: dict) -> dict:
        assert self.scripted_chunk is not None, "test must set scripted_chunk first"
        self.last_request = request
        return {"actions": self.scripted_chunk}


@pytest.fixture
def upstream_and_ours():
    """Construct both clients sharing the same scripted websocket client."""
    with mock.patch(
        "robolab.inference.pi0_jointvel.websocket_client_policy.WebsocketClientPolicy",
        new=_ScriptedWebsocketClient,
    ):
        from robolab.inference.pi0_jointvel import Pi0DroidJointvelClient

        upstream = Pi0DroidJointvelClient(
            remote_host="localhost", remote_port=9999, open_loop_horizon=8,
        )
        return upstream


def _fake_sim_obs(joint_pos: np.ndarray, gripper: float) -> dict:
    """Shape matches what RoboLab's env.step() returns — what both our code
    and upstream's _extract_observation consume."""
    ext = torch.zeros((1, 180, 320, 3), dtype=torch.uint8)
    wrist = torch.zeros((1, 180, 320, 3), dtype=torch.uint8)
    return {
        "image_obs": {"external_cam": ext, "wrist_cam": wrist},
        "proprio_obs": {
            "arm_joint_pos": torch.from_numpy(joint_pos[None, :]).float(),
            "gripper_pos": torch.tensor([[gripper]], dtype=torch.float32),
        },
    }


def _our_stream(raw_chunk: np.ndarray, joint_pos_at_anchor: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Walk our main_robolab inner loop's action processing for n_steps,
    using the same raw_chunk the upstream client receives. Returns the list
    of 8-dim actions we'd hand to env.step()."""
    q_target = joint_pos_at_anchor.astype(np.float32).copy()
    emitted = []
    for i in range(n_steps):
        raw = raw_chunk[i]
        q_target = mr._integrate_velocity(raw[:7], q_target)
        gripper = 1.0 if raw[-1].item() > 0.5 else 0.0
        action = np.concatenate([q_target, np.array([gripper], dtype=np.float32)])
        emitted.append(action)
    return emitted


def _upstream_stream(upstream_client, raw_chunk: np.ndarray, joint_pos_at_anchor: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Call upstream.infer() n_steps times. After the first call the chunk
    is cached internally (open_loop_horizon=8), so subsequent calls reuse
    the same integrator state — same as our loop's behaviour within a chunk."""
    upstream_client.reset()
    upstream_client.client.scripted_chunk = raw_chunk.astype(np.float32)

    emitted = []
    for _ in range(n_steps):
        obs = _fake_sim_obs(joint_pos_at_anchor, gripper=0.0)
        result = upstream_client.infer(obs, instruction="test", env_id=0)
        emitted.append(np.asarray(result["action"], dtype=np.float32))
    return emitted


# -- Parity tests ---------------------------------------------------------


def test_zero_velocity_holds_position(upstream_and_ours):
    """With v=0 for every step, both implementations should emit
    action = [joint_pos_at_anchor, 0_gripper] verbatim, identical."""
    upstream = upstream_and_ours
    joint_pos = np.array([0.0, -0.5, 0.1, -1.5, 0.0, 1.5, 0.0], dtype=np.float32)
    raw = np.zeros((16, 8), dtype=np.float32)

    ours = _our_stream(raw, joint_pos, n_steps=8)
    theirs = _upstream_stream(upstream, raw, joint_pos, n_steps=8)

    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0, err_msg=f"mismatch at step {i}")


def test_constant_positive_velocity_integrates_identically(upstream_and_ours):
    """v_norm = +1 on every joint; both should emit the same q_target on each step."""
    upstream = upstream_and_ours
    joint_pos = np.zeros(7, dtype=np.float32)
    raw = np.zeros((16, 8), dtype=np.float32)
    raw[:, :7] = 1.0

    ours = _our_stream(raw, joint_pos, n_steps=8)
    theirs = _upstream_stream(upstream, raw, joint_pos, n_steps=8)

    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0, err_msg=f"step {i} | ours={a} theirs={b}")


def test_out_of_range_velocity_clipped_identically(upstream_and_ours):
    """v_norm values outside [-1, 1] must be clipped by both implementations."""
    upstream = upstream_and_ours
    joint_pos = np.array([0.1] * 7, dtype=np.float32)
    raw = np.zeros((16, 8), dtype=np.float32)
    # Alternating +5 / -5 on different joints
    raw[:, 0] = 5.0
    raw[:, 1] = -5.0
    raw[:, 2] = 0.7

    ours = _our_stream(raw, joint_pos, n_steps=8)
    theirs = _upstream_stream(upstream, raw, joint_pos, n_steps=8)

    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0, err_msg=f"step {i}")


@pytest.mark.parametrize("raw_gripper, expected", [(0.0, 0.0), (0.49, 0.0), (0.5, 0.0), (0.51, 1.0), (1.0, 1.0)])
def test_gripper_binarization_matches(upstream_and_ours, raw_gripper, expected):
    upstream = upstream_and_ours
    joint_pos = np.zeros(7, dtype=np.float32)
    raw = np.zeros((16, 8), dtype=np.float32)
    raw[:, -1] = raw_gripper

    ours = _our_stream(raw, joint_pos, n_steps=1)
    theirs = _upstream_stream(upstream, raw, joint_pos, n_steps=1)

    assert ours[0][-1] == expected
    assert theirs[0][-1] == expected
    np.testing.assert_allclose(ours[0], theirs[0], atol=0, rtol=0)


def test_random_velocity_stream_matches_bitwise(upstream_and_ours):
    """Randomized 8-step chunk; integrated actions must be bit-identical."""
    upstream = upstream_and_ours
    rng = np.random.default_rng(42)
    joint_pos = rng.uniform(-2.0, 2.0, size=7).astype(np.float32)
    raw = rng.uniform(-1.2, 1.2, size=(16, 8)).astype(np.float32)
    raw[:, -1] = rng.uniform(0.0, 1.0, size=16).astype(np.float32)

    ours = _our_stream(raw, joint_pos, n_steps=8)
    theirs = _upstream_stream(upstream, raw, joint_pos, n_steps=8)

    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0, err_msg=f"step {i}")


def test_constants_identical_to_upstream(upstream_and_ours):
    """Our DROID_VEL_LIMITS / dt must match whatever values upstream uses."""
    upstream = upstream_and_ours
    np.testing.assert_allclose(mr.DROID_VEL_LIMITS, upstream.vel_limits)
    assert (1.0 / mr.SIM_CONTROL_FREQUENCY) == pytest.approx(upstream.dt, abs=0)
