"""Unit tests for ``main_mpc._load_extrinsics`` — the calibration-JSON
loader that real-robot MPC uses to resolve the ZED ``camera_in_base``
6-vec.

Covers:
  * accepts both DROID-native ``{"pose": [...]}`` and eva_pal_share
    ``{"extrinsics": [...]}`` field names
  * respects the search-path order (first existing file wins)
  * honors an explicit override-path argument
  * raises FileNotFoundError when no file has a usable entry — we
    removed the hardcoded fallback so misconfigured robots fail loudly
  * tolerates non-dict entries / missing keys / wrong-length vectors
    by continuing down the search order rather than crashing
  * prints the resolved file + timestamp so operators see which
    calibration was actually picked
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# main_mpc imports torch lazily inside _refine_chunk_with_mpc but not at
# module level. So this import should work on the sim venv.
import main_mpc  # noqa: E402


# ----------------------------------------------------------------------
def _make_json(tmp: Path, data: dict, name="calibration_info.json") -> str:
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ----------------------------------------------------------------------
def test_loads_droid_native_pose_field(tmp_path):
    """DROID writes entries with the key 'pose'. Loader must accept it."""
    pose = [-0.037, 0.642, 0.762, -2.05, -0.003, -2.32]
    path = _make_json(tmp_path, {
        "26368109_left": {"pose": pose, "timestamp": time.time()},
    })
    ext = main_mpc._load_extrinsics(override_path=path)
    assert ext.shape == (6,) and ext.dtype == np.float32
    np.testing.assert_allclose(ext.astype(np.float64), pose, atol=1e-6)


def test_loads_eva_pal_share_extrinsics_field(tmp_path):
    """The older eva_pal_share file stores the same 6-vec under 'extrinsics'."""
    pose = [0.1, -0.02, 0.25, -1.53, 0.02, -1.56]
    path = _make_json(tmp_path, {
        "26368109_left": {"extrinsics": pose, "timestamp": time.time()},
    })
    ext = main_mpc._load_extrinsics(override_path=path)
    np.testing.assert_allclose(ext.astype(np.float64), pose, atol=1e-6)


def test_prefers_pose_over_extrinsics_when_both_present(tmp_path):
    """If both keys are present (unlikely but possible during format transition)
    the DROID-style 'pose' takes precedence."""
    path = _make_json(tmp_path, {
        "26368109_left": {
            "pose": [1.0, 0.0, 0.0, 0, 0, 0],
            "extrinsics": [2.0, 0.0, 0.0, 0, 0, 0],
            "timestamp": time.time(),
        },
    })
    ext = main_mpc._load_extrinsics(override_path=path)
    assert ext[0] == pytest.approx(1.0)


def test_raises_when_no_calibration_file_found(tmp_path):
    """No fallback; a misconfigured rig fails loudly at launch."""
    path = str(tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError) as exc:
        main_mpc._load_extrinsics(override_path=path)
    assert "26368109_left" in str(exc.value)
    assert "calibration" in str(exc.value).lower()


def test_raises_when_camera_key_missing(tmp_path):
    """File exists but our camera serial isn't in it → fail loudly."""
    path = _make_json(tmp_path, {
        "OTHER_SERIAL_left": {"pose": [0, 0, 0, 0, 0, 0]},
    })
    with pytest.raises(FileNotFoundError):
        main_mpc._load_extrinsics(override_path=path)


def test_raises_on_wrong_length_vector(tmp_path):
    """6-vec required — 3-vec or 7-vec must not be silently accepted."""
    path = _make_json(tmp_path, {
        "26368109_left": {"pose": [1, 2, 3], "timestamp": time.time()},
    })
    with pytest.raises(FileNotFoundError):  # propagates as no-usable-entry
        main_mpc._load_extrinsics(override_path=path)


def test_raises_on_unparseable_json(tmp_path):
    """Corrupt JSON should propagate as no-usable-entry (skipped, not crash)."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        main_mpc._load_extrinsics(override_path=str(p))


def test_respects_search_path_order_first_existing_wins(tmp_path, monkeypatch):
    """Priority: laptop aurora > laptop droid_p2r > repo > eva_pal_share. The
    first existing file wins even if a lower-priority one has fresher data."""
    hi = _make_json(tmp_path, {
        "26368109_left": {"pose": [1, 0, 0, 0, 0, 0]},
    }, name="first.json")
    lo = _make_json(tmp_path, {
        "26368109_left": {"pose": [2, 0, 0, 0, 0, 0]},
    }, name="second.json")
    monkeypatch.setattr(main_mpc, "_CALIBRATION_JSON_SEARCH_PATHS",
                         (hi, lo))
    ext = main_mpc._load_extrinsics()
    assert ext[0] == pytest.approx(1.0)


def test_skips_missing_files_in_search_path(tmp_path, monkeypatch):
    """When the first few paths don't exist, loader walks to the next."""
    ghost1 = str(tmp_path / "ghost1.json")  # not created
    ghost2 = str(tmp_path / "ghost2.json")  # not created
    real = _make_json(tmp_path, {
        "26368109_left": {"pose": [99, 0, 0, 0, 0, 0]},
    }, name="real.json")
    monkeypatch.setattr(main_mpc, "_CALIBRATION_JSON_SEARCH_PATHS",
                         (ghost1, ghost2, real))
    ext = main_mpc._load_extrinsics()
    assert ext[0] == pytest.approx(99.0)


def test_skips_files_without_camera_entry(tmp_path, monkeypatch):
    """Even if a JSON exists, loader skips it if our key isn't present."""
    wrong = _make_json(tmp_path, {"OTHER_SERIAL": {"pose": [0] * 6}}, name="wrong.json")
    right = _make_json(tmp_path, {
        "26368109_left": {"pose": [42, 0, 0, 0, 0, 0]},
    }, name="right.json")
    monkeypatch.setattr(main_mpc, "_CALIBRATION_JSON_SEARCH_PATHS",
                         (wrong, right))
    ext = main_mpc._load_extrinsics()
    assert ext[0] == pytest.approx(42.0)


def test_override_path_bypasses_search(tmp_path, monkeypatch):
    """When override_path is passed, the configured search list is ignored —
    even if the override file is missing/bad we don't silently fall back."""
    # Poison the search paths with a working file. It should NOT be used
    # when an override is given.
    poison = _make_json(tmp_path, {
        "26368109_left": {"pose": [100, 0, 0, 0, 0, 0]},
    }, name="poison.json")
    monkeypatch.setattr(main_mpc, "_CALIBRATION_JSON_SEARCH_PATHS", (poison,))

    # Valid override
    good = _make_json(tmp_path, {
        "26368109_left": {"pose": [7, 0, 0, 0, 0, 0]},
    }, name="good.json")
    ext = main_mpc._load_extrinsics(override_path=good)
    assert ext[0] == pytest.approx(7.0)

    # Missing override
    with pytest.raises(FileNotFoundError):
        main_mpc._load_extrinsics(override_path=str(tmp_path / "not_here.json"))


def test_non_default_camera_key(tmp_path):
    """The loader accepts a camera_key argument for the wrist / other ZEDs."""
    path = _make_json(tmp_path, {
        "15512737_left": {"pose": [-0.07, 0.03, 0.02, -0.34, 0.01, -1.55],
                           "timestamp": time.time()},
    })
    ext = main_mpc._load_extrinsics(override_path=path, camera_key="15512737_left")
    assert ext.shape == (6,)


def test_prints_resolved_file_and_timestamp(tmp_path, capsys):
    """Must log which calibration file + date got picked so the operator
    can see drift at launch-time."""
    path = _make_json(tmp_path, {
        "26368109_left": {"pose": [0] * 6, "timestamp": 1700000000.0},
    })
    main_mpc._load_extrinsics(override_path=path)
    out = capsys.readouterr().out
    assert "loaded 26368109_left extrinsics from" in out
    assert path in out
    # ISO timestamp for 1700000000.0 should include the year in the string.
    assert "2023" in out


def test_repo_committed_calibration_is_valid():
    """End-to-end: the repo-tracked calibration file is actually usable by
    the loader. Guards against someone committing a malformed file."""
    ext = main_mpc._load_extrinsics(override_path=str(
        _REPO / "calibration" / "calibration_info.json"
    ))
    assert ext.shape == (6,)
    assert np.all(np.isfinite(ext))
