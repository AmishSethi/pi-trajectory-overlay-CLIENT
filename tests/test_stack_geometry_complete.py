"""Unit tests for ``main._stack_geometry_complete`` — the pixel-space
spatial heuristic that OR-supplements Gemini's textual
``query_step_completion`` so stack/place/put sub-steps advance once the
manipulating-object pixel sits directly above the target-object pixel.

Covers:
  * returns True for canonical stacked configuration (manip directly above
    target within the threshold box)
  * returns False when pixel geometry is wrong (too far horizontally, too
    far vertically, manip BELOW target)
  * returns False for non-stacking verbs (no 'stack'/'place'/'put')
  * returns False when detections are missing for either object
  * tolerates loose key spelling in ``_last_object_locations`` via
    case-insensitive substring match (e.g. Gemini returns ``'red cup'``
    when the step says ``'Red cup'``)
  * returns False when both objects are at the same pixel (dy=0)
  * does NOT suppress Gemini's existing yes answer (tested by verifying
    _check_step_completion OR-logic works end-to-end with a stub)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# We need to import main.py but main.py has heavy imports (openpi_client,
# trajectory_predictor, google-genai, etc.) that aren't all available on the
# GPU box's sim venv. Stub the heavy ones before import so pytest runs green.
# NOTE: any symbol used in an annotation (e.g. `TraceOverlayConfig | None`)
# must be stubbed as a CLASS, not a lambda/function, because Python 3.10+
# evaluates `X | None` at module-load time which requires X to be a type.
def _stub_module(name: str, functions: list[str] = (), classes: list[str] = ()):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for f in functions:
        setattr(mod, f, lambda *a, **k: None)
    for c in classes:
        setattr(mod, c, type(c, (), {}))
    sys.modules[name] = mod
    return mod


_stub_module("openpi_client")
_stub_module("openpi_client.image_tools", functions=["resize_with_pad"])
_stub_module("openpi_client.websocket_client_policy", classes=["WebsocketClientPolicy"])
_stub_module("trajectory_predictor",
              functions=["encode_pil_image", "query_step_completion",
                         "query_target_location", "query_target_objects",
                         "query_trajectory", "rescale_trajectory",
                         "resize_for_api"])
_stub_module("traj_vis_utils",
              functions=["add_trace_overlay"],
              classes=["TraceOverlayConfig"])
_stub_module("inference_visualizer",
              classes=["InferenceVisualizer", "InferenceInfo"])

import main as client_main   # noqa: E402


# ----------------------------------------------------------------------
class _FakePlanning:
    """Minimal stand-in for main.PlanningState used by
    _stack_geometry_complete. Only needs _last_object_locations."""
    def __init__(self, locs):
        self._last_object_locations = locs


def _step(text, manip="red cup", target="green cup"):
    return {"step": text, "manipulating_object": manip, "target_related_object": target}


# ----------------------------------------------------------------------
#                    _stack_geometry_complete unit tests
# ----------------------------------------------------------------------
def test_hits_on_canonical_stacked_geometry():
    """Manipulating object directly above target (small dx, small positive dy)
    → return True. Same numbers we saw in the failed cup-stack run at t=300."""
    locs = {"red cup": (543.0, 345.0), "green cup": (543.0, 357.0)}
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is True


def test_misses_when_horizontally_off():
    """If |dx| > 40 px the manip is in a different X column → not stacked."""
    locs = {"red cup": (500.0, 345.0), "green cup": (550.0, 357.0)}  # dx=50
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_when_manip_below_target():
    """Manipulator y is greater than target y (visually below) → not stacked."""
    locs = {"red cup": (543.0, 400.0), "green cup": (543.0, 357.0)}  # dy = -43
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_when_vertical_gap_too_large():
    """|dy| > 60 px means the two objects aren't touching/stacked."""
    locs = {"red cup": (543.0, 200.0), "green cup": (543.0, 357.0)}  # dy = +157
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_at_zero_vertical_gap():
    """dy = 0 exactly is not a stack (the objects are at the same image y
    — probably occluded or side-by-side). Heuristic requires dy > 0."""
    locs = {"red cup": (543.0, 357.0), "green cup": (543.0, 357.0)}
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_without_stacking_verb():
    """Text doesn't contain stack/place/put → heuristic bypasses without
    returning True even when geometry would otherwise match."""
    locs = {"red cup": (543.0, 345.0), "green cup": (543.0, 357.0)}
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Pick up the red cup")) is False
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Pour from the red cup to the green cup")) is False
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Move the red cup near the green cup")) is False


def test_hits_on_all_three_stacking_verbs():
    """stack / place / put all trigger the heuristic."""
    locs = {"red cup": (543.0, 345.0), "green cup": (543.0, 357.0)}
    for verb in ("Stack", "Place", "Put"):
        step = _step(f"{verb} the red cup in the green cup")
        assert client_main._stack_geometry_complete(_FakePlanning(locs), step) is True


def test_misses_when_manipulating_object_not_detected():
    locs = {"green cup": (543.0, 357.0)}  # no red cup
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_when_target_object_not_detected():
    locs = {"red cup": (543.0, 345.0)}  # no green cup
    assert client_main._stack_geometry_complete(_FakePlanning(locs),
                                                 _step("Stack the red cup in the green cup")) is False


def test_misses_when_locations_dict_is_empty_or_missing():
    assert client_main._stack_geometry_complete(_FakePlanning({}),
                                                 _step("Stack the red cup in the green cup")) is False
    # Simulate PlanningState that never had _predict_trajectory run (attr missing):
    class _NoAttr:
        pass
    assert client_main._stack_geometry_complete(_NoAttr(),
                                                 _step("Stack the red cup in the green cup")) is False


def test_loose_name_match_case_insensitive_substring():
    """Gemini often returns key 'red cup' even when the step says 'Red Cup'
    or 'a red cup'. The helper substrings-matches, case-insensitive."""
    locs = {"red cup": (543.0, 345.0), "the green cup": (543.0, 357.0)}
    step = {"step": "Stack the Red Cup in The Green Cup",
            "manipulating_object": "Red Cup", "target_related_object": "Green Cup"}
    assert client_main._stack_geometry_complete(_FakePlanning(locs), step) is True


def test_misses_when_step_data_missing_object_names():
    """manipulating_object or target_related_object empty → can't match."""
    locs = {"red cup": (543.0, 345.0), "green cup": (543.0, 357.0)}
    # empty manip
    step_no_manip = {"step": "Stack something in the green cup",
                     "manipulating_object": "", "target_related_object": "green cup"}
    assert client_main._stack_geometry_complete(_FakePlanning(locs), step_no_manip) is False
    step_no_target = {"step": "Stack the red cup somewhere",
                      "manipulating_object": "red cup", "target_related_object": ""}
    assert client_main._stack_geometry_complete(_FakePlanning(locs), step_no_target) is False


# ----------------------------------------------------------------------
#               _check_step_completion OR-logic integration
# ----------------------------------------------------------------------
class _FakePlanningFull:
    """Adds img_history + current_step so _check_step_completion is exercised."""
    def __init__(self, locs, step_data):
        self._last_object_locations = locs
        self.img_history = ["stub_image"]  # non-empty so the helper proceeds
        self.current_step = step_data
        self.step_idx = 1


class _Args:
    gemini_model = "stub"


def test_check_returns_true_when_gemini_yes_heuristic_no(monkeypatch):
    def _fake_qsc(*a, **k):
        return {"is_complete": True}
    monkeypatch.setattr(client_main, "query_step_completion", _fake_qsc)
    # geometry does NOT match
    pln = _FakePlanningFull({"red cup": (100.0, 400.0), "green cup": (543.0, 357.0)},
                             _step("Stack the red cup in the green cup"))
    assert client_main._check_step_completion(pln, _Args()) is True


def test_check_returns_true_when_gemini_no_heuristic_yes(monkeypatch, capsys):
    def _fake_qsc(*a, **k):
        return {"is_complete": False}
    monkeypatch.setattr(client_main, "query_step_completion", _fake_qsc)
    pln = _FakePlanningFull({"red cup": (543.0, 345.0), "green cup": (543.0, 357.0)},
                             _step("Stack the red cup in the green cup"))
    assert client_main._check_step_completion(pln, _Args()) is True
    # Verify the heuristic override prints a log line so operators see the bypass.
    out = capsys.readouterr().out
    assert "spatial heuristic" in out and "COMPLETE" in out


def test_check_returns_false_when_both_say_no(monkeypatch):
    def _fake_qsc(*a, **k):
        return {"is_complete": False}
    monkeypatch.setattr(client_main, "query_step_completion", _fake_qsc)
    # geometry does NOT match
    pln = _FakePlanningFull({"red cup": (100.0, 400.0), "green cup": (543.0, 357.0)},
                             _step("Stack the red cup in the green cup"))
    assert client_main._check_step_completion(pln, _Args()) is False


def test_check_short_circuits_when_no_img_history(monkeypatch):
    """Without img_history we never call Gemini — guard against nonsense."""
    called = {"n": 0}
    def _fake_qsc(*a, **k):
        called["n"] += 1
        return {"is_complete": True}
    monkeypatch.setattr(client_main, "query_step_completion", _fake_qsc)
    pln = _FakePlanningFull({"red cup": (543, 345), "green cup": (543, 357)},
                             _step("Stack the red cup in the green cup"))
    pln.img_history = []
    assert client_main._check_step_completion(pln, _Args()) is False
    assert called["n"] == 0
