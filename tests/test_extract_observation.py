"""Tests for _extract_observation in main.py.

Covers the "right camera optional when external=left" behavior and the
upgraded error message — both uncommitted changes that the existing suite
doesn't cover.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.expanduser("~/pi-trajectory-overlay")
sys.path.insert(0, REPO)

# Importing main.py pulls in heavy DROID/openpi deps. Skip if not available
# (though in eva_jiani_env they should be).
try:
    from main import _extract_observation
    MAIN_OK = True
    _reason = ""
except Exception as e:
    MAIN_OK = False
    _reason = f"{type(e).__name__}: {e}"

pytestmark = pytest.mark.skipif(not MAIN_OK, reason=f"main.py import failed: {_reason}")


class _FakeArgs:
    left_camera_id = "26368109"
    right_camera_id = "25455306"
    wrist_camera_id = "15512737"
    external_camera = "left"


def _bgra_image(h=90, w=160, color_bgra=(10, 20, 30, 255)):
    """Simulate a BGRA frame like ZED returns."""
    im = np.zeros((h, w, 4), dtype=np.uint8)
    im[:] = color_bgra
    return im


def _robot_state(joints=None, gripper=0.0):
    """Build the minimal robot_state dict the DROID env emits.

    The real dict uses plural 'joint_positions' and singular 'gripper_position'.
    Any divergence in schema between DROID versions would break _extract_observation
    at the `joint_positions`/`gripper_position` lookups.
    """
    if joints is None:
        joints = [0.0] * 7
    return {"joint_positions": list(joints), "gripper_position": float(gripper)}


class TestExternalLeft:
    """When external_camera='left', only left + wrist are required."""

    def test_missing_right_is_fine(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "26368109_left": _bgra_image(color_bgra=(1, 2, 3, 255)),
                # right intentionally absent
                "15512737_left": _bgra_image(color_bgra=(7, 8, 9, 255)),
            },
            "robot_state": _robot_state(),
        }
        out = _extract_observation(args, obs)
        assert out["left_image"].shape == (90, 160, 3)
        assert out["wrist_image"].shape == (90, 160, 3)
        assert out.get("right_image") is None

    def test_bgr_to_rgb_conversion(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "26368109_left": _bgra_image(color_bgra=(1, 2, 3, 255)),  # B,G,R,A
                "15512737_left": _bgra_image(),
            },
            "robot_state": _robot_state(),
        }
        out = _extract_observation(args, obs)
        # BGR->RGB means source pixel (1,2,3) becomes (3,2,1).
        assert tuple(out["left_image"][0, 0]) == (3, 2, 1)

    def test_alpha_dropped(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "26368109_left": _bgra_image(color_bgra=(1, 2, 3, 255)),
                "15512737_left": _bgra_image(),
            },
            "robot_state": _robot_state(),
        }
        out = _extract_observation(args, obs)
        assert out["left_image"].shape[-1] == 3
        assert out["wrist_image"].shape[-1] == 3


class TestExternalRight:
    """When external_camera='right', only right + wrist are required."""

    def test_missing_left_is_fine(self):
        args = _FakeArgs()
        args.external_camera = "right"
        obs = {
            "image": {
                "25455306_left": _bgra_image(color_bgra=(4, 5, 6, 255)),  # right
                "15512737_left": _bgra_image(),                          # wrist
            },
            "robot_state": _robot_state(),
        }
        out = _extract_observation(args, obs)
        assert out["right_image"].shape == (90, 160, 3)
        assert tuple(out["right_image"][0, 0]) == (6, 5, 4)  # BGR->RGB
        assert out.get("left_image") is None


class TestMissingCameras:
    """Missing required cameras must raise a useful error."""

    def test_missing_wrist_raises_with_details(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "26368109_left": _bgra_image(),  # only left; wrist missing
            },
            "robot_state": _robot_state(),
        }
        with pytest.raises(RuntimeError) as exc:
            _extract_observation(args, obs)
        msg = str(exc.value)
        assert "wrist" in msg
        assert "15512737" in msg
        assert "Available image keys" in msg

    def test_missing_external_left_raises(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "15512737_left": _bgra_image(),  # only wrist; left missing
            },
            "robot_state": _robot_state(),
        }
        with pytest.raises(RuntimeError) as exc:
            _extract_observation(args, obs)
        assert "left" in str(exc.value)

    def test_missing_image_key_entirely(self):
        args = _FakeArgs()
        obs = {"robot_state": _robot_state()}
        with pytest.raises(RuntimeError) as exc:
            _extract_observation(args, obs)
        assert "Available image keys" in str(exc.value)

    def test_empty_image_dict(self):
        args = _FakeArgs()
        obs = {"image": {}, "robot_state": _robot_state()}
        with pytest.raises(RuntimeError):
            _extract_observation(args, obs)


class TestOutputStructure:
    """Verify robot_state fields are present under the expected keys."""

    def test_robot_state_fields_extracted(self):
        args = _FakeArgs()
        args.external_camera = "left"
        obs = {
            "image": {
                "26368109_left": _bgra_image(),
                "15512737_left": _bgra_image(),
            },
            "robot_state": _robot_state(joints=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], gripper=0.25),
        }
        out = _extract_observation(args, obs)
        assert "joint_position" in out, f"keys: {list(out.keys())}"
        assert "gripper_position" in out, f"keys: {list(out.keys())}"
        np.testing.assert_allclose(out["joint_position"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        # gripper_position is wrapped in a length-1 array in main.py
        np.testing.assert_allclose(out["gripper_position"], [0.25])
