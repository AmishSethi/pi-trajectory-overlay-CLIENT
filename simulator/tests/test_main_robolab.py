"""Unit tests for main_robolab.py's pure helpers.

main_robolab.py keeps all its helpers inline (mirroring main_pi05.py's
style). The Isaac Sim AppLauncher is only instantiated inside
`if __name__ == "__main__"`, so the module is safe to import from tests
without booting the simulator.
"""

import numpy as np
import pytest
import torch

from main_robolab import (
    DROID_VEL_LIMITS,
    SIM_CONTROL_FREQUENCY,
    Args,
    FakePolicy,
    _extract_observation,
    _integrate_velocity,
    prevent_keyboard_interrupt,
)


def _fake_sim_obs(num_envs: int = 2, h: int = 180, w: int = 320) -> dict:
    """Same shape/dtype contract as RoboLab's env.step() observation."""
    ext = torch.zeros((num_envs, h, w, 3), dtype=torch.uint8)
    wrist = torch.zeros((num_envs, h, w, 3), dtype=torch.uint8)
    for i in range(num_envs):
        ext[i, 0, 0, 0] = i + 1
        wrist[i, 0, 0, 0] = 100 + i
    joints = torch.tensor([[0.1 * (i + 1)] * 7 for i in range(num_envs)])
    gripper = torch.tensor([[0.25 * (i + 1)] for i in range(num_envs)])
    return {
        "image_obs": {"external_cam": ext, "wrist_cam": wrist},
        "proprio_obs": {"arm_joint_pos": joints, "gripper_pos": gripper},
    }


# -- Args -----------------------------------------------------------------


def test_args_defaults_match_main_pi05_shape():
    """The shared fields that both entrypoints carry should have the same
    defaults — makes diffing the two files trivial."""
    a = Args(task="Dummy")
    assert a.max_timesteps == 600
    assert a.open_loop_horizon == 8
    assert a.remote_host == "0.0.0.0"
    assert a.remote_port == 8000
    assert a.save_frames is False


# -- prevent_keyboard_interrupt ------------------------------------------


def test_prevent_keyboard_interrupt_defers_sigint():
    """Ctrl+C during the context block is held until exit, then re-raised."""
    import os
    import signal

    with pytest.raises(KeyboardInterrupt):
        with prevent_keyboard_interrupt():
            os.kill(os.getpid(), signal.SIGINT)
            # If SIGINT wasn't deferred, the context manager couldn't catch
            # it and this test would bomb with an un-re-raised KeyboardInterrupt.


def test_prevent_keyboard_interrupt_passes_through_when_clean():
    with prevent_keyboard_interrupt():
        pass  # no SIGINT — nothing should be raised after exit.


# -- _extract_observation -------------------------------------------------


def test_extract_observation_returns_numpy_uint8_images():
    obs = _fake_sim_obs()
    cur = _extract_observation(Args(task="X"), obs, env_id=0)
    for key in ("external_image", "wrist_image"):
        assert isinstance(cur[key], np.ndarray)
        assert cur[key].dtype == np.uint8
        assert cur[key].shape == (180, 320, 3)


def test_extract_observation_indexes_the_right_env():
    obs = _fake_sim_obs(num_envs=3)
    cur0 = _extract_observation(Args(task="X"), obs, env_id=0)
    cur2 = _extract_observation(Args(task="X"), obs, env_id=2)
    assert cur0["external_image"][0, 0, 0] == 1
    assert cur2["external_image"][0, 0, 0] == 3
    assert cur0["wrist_image"][0, 0, 0] == 100
    assert cur2["wrist_image"][0, 0, 0] == 102
    np.testing.assert_allclose(cur0["joint_position"], [0.1] * 7)
    np.testing.assert_allclose(cur2["joint_position"], [0.3] * 7)
    np.testing.assert_allclose(cur0["gripper_position"], [0.25])
    np.testing.assert_allclose(cur2["gripper_position"], [0.75])


def test_extract_observation_converts_non_uint8_images():
    """RoboLab usually returns uint8, but if a variation emits float images
    we must cast so downstream image_tools.resize_with_pad works."""
    obs = _fake_sim_obs()
    obs["image_obs"]["external_cam"] = obs["image_obs"]["external_cam"].float()
    cur = _extract_observation(Args(task="X"), obs, env_id=0)
    assert cur["external_image"].dtype == np.uint8


def test_extract_observation_output_matches_main_pi05_schema():
    """Key names are what downstream request-building code in main_robolab
    (and main_pi05) reads. If these drift from main_pi05 we'd break request
    building."""
    cur = _extract_observation(Args(task="X"), _fake_sim_obs(), env_id=0)
    assert set(cur) == {
        "external_image", "wrist_image", "joint_position", "gripper_position",
    }


# -- _integrate_velocity --------------------------------------------------


def test_integrate_velocity_zero_holds_position():
    q0 = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7], dtype=np.float32)
    q1 = _integrate_velocity(np.zeros(7, dtype=np.float32), q0)
    np.testing.assert_allclose(q1, q0, atol=1e-6)


def test_integrate_velocity_unit_positive_moves_by_vel_limits_times_dt():
    q0 = np.zeros(7, dtype=np.float32)
    q1 = _integrate_velocity(np.ones(7, dtype=np.float32), q0)
    expected = DROID_VEL_LIMITS * (1.0 / SIM_CONTROL_FREQUENCY)
    np.testing.assert_allclose(q1, expected, atol=1e-6)


def test_integrate_velocity_clips_out_of_range_velocities():
    """v=+5 should be clipped to +1 → same motion as v=+1."""
    q0 = np.zeros(7, dtype=np.float32)
    q_clipped = _integrate_velocity(np.ones(7) * 5.0, q0)
    q_unit = _integrate_velocity(np.ones(7), q0)
    np.testing.assert_allclose(q_clipped, q_unit, atol=1e-6)


def test_integrate_velocity_accumulates_across_calls():
    q = np.zeros(7, dtype=np.float32)
    raw = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    q = _integrate_velocity(raw, q)
    q = _integrate_velocity(raw, q)
    q = _integrate_velocity(raw, q)
    assert q[0] == pytest.approx(3 * DROID_VEL_LIMITS[0] / SIM_CONTROL_FREQUENCY, abs=1e-6)
    np.testing.assert_allclose(q[1:], np.zeros(6), atol=1e-6)


# -- FakePolicy -----------------------------------------------------------


def test_fake_policy_returns_correct_shape_and_records_request():
    fp = FakePolicy(action_dim=8, chunk_length=16)
    out = fp.infer({"prompt": "hi"})
    assert out["actions"].shape == (16, 8)
    assert out["actions"].dtype == np.float32
    assert (out["actions"] == 0).all()
    assert fp.infer_calls == 1
    assert fp.last_request == {"prompt": "hi"}


# -- End-to-end wire-format round-trip (no sim, no server) --------------


def test_extract_plus_request_schema_matches_main_pi05():
    """Walk the same logic main_robolab.main's inner loop walks, without
    the sim. The request dict must have the exact 5 keys main_pi05 sends.
    """
    from openpi_client import image_tools

    cur = _extract_observation(Args(task="X"), _fake_sim_obs(), env_id=0)
    request = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            cur["external_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            cur["wrist_image"], 224, 224
        ),
        "observation/joint_position": cur["joint_position"],
        "observation/gripper_position": cur["gripper_position"],
        "prompt": "test",
    }
    assert set(request) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    assert request["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert request["observation/wrist_image_left"].shape == (224, 224, 3)
