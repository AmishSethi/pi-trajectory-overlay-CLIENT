"""Tests for the plan-once step-merging logic in main.py.

When --max-plan-count=1 and GPT decomposes the instruction into >1 step, the
planner merges first+last into ONE synthetic step that uses the full user
instruction, so GPT plans a trajectory that covers the WHOLE task (pick->place)
rather than just the first step (pick).

The merging is inline in main.py so these tests replicate the logic verbatim
and verify it behaves as specified across edge cases. If main.py changes these
tests should be re-synced.
"""
import pytest


def _merge_plan_once(steps, instruction):
    """Replicates the inline merging logic in main.py.

    Keep in sync with the corresponding block in main.py (search for
    '[plan-once] Merging').
    """
    if len(steps) <= 1:
        return list(steps)
    first = steps[0]
    last = steps[-1]
    merged = {
        "step": instruction,
        "manipulating_object": first["manipulating_object"],
        "target_related_object": (
            last.get("target_related_object")
            or first.get("target_related_object", "")
        ),
        "target_location": (
            last.get("target_location")
            or first.get("target_location", "")
        ),
    }
    return [merged]


class TestMergeTwoStepPickAndPlace:
    def test_merges_manipulate_from_first_target_from_last(self):
        steps = [
            {
                "step": "Pick up the marker",
                "manipulating_object": "marker",
                "target_related_object": "",
                "target_location": "",
            },
            {
                "step": "Put the marker in the bowl",
                "manipulating_object": "marker",
                "target_related_object": "bowl",
                "target_location": "in the bowl",
            },
        ]
        instruction = "Pick up the marker and put it in the bowl"
        out = _merge_plan_once(steps, instruction)
        assert len(out) == 1
        assert out[0]["step"] == instruction
        assert out[0]["manipulating_object"] == "marker"
        assert out[0]["target_related_object"] == "bowl"
        assert out[0]["target_location"] == "in the bowl"


class TestMergeThreeStep:
    def test_takes_first_manipulate_and_last_target(self):
        steps = [
            {"step": "Pick up block", "manipulating_object": "block", "target_related_object": "", "target_location": ""},
            {"step": "Move block over bowl", "manipulating_object": "block", "target_related_object": "bowl", "target_location": "over the bowl"},
            {"step": "Lower block into bowl", "manipulating_object": "block", "target_related_object": "bowl", "target_location": "in the bowl"},
        ]
        instruction = "Pick up the block and put it in the bowl"
        out = _merge_plan_once(steps, instruction)
        assert len(out) == 1
        assert out[0]["manipulating_object"] == "block"
        assert out[0]["target_location"] == "in the bowl"


class TestPassthrough:
    def test_single_step_unchanged(self):
        steps = [{
            "step": "Press the red button",
            "manipulating_object": "red button",
            "target_related_object": "",
            "target_location": "",
        }]
        instruction = "Press the red button"
        out = _merge_plan_once(steps, instruction)
        assert len(out) == 1
        assert out[0]["step"] == "Press the red button"  # NOT replaced
        assert out[0] is steps[0]  # reference preserved

    def test_empty_steps_unchanged(self):
        out = _merge_plan_once([], "some instruction")
        assert out == []


class TestFallbackToFirstStep:
    """When last step lacks target_related_object, fall back to first step's."""

    def test_falls_back_when_last_missing_target_related(self):
        steps = [
            {"step": "Pick up marker next to bowl", "manipulating_object": "marker", "target_related_object": "bowl", "target_location": "next to bowl"},
            {"step": "Now move it", "manipulating_object": "marker", "target_related_object": "", "target_location": ""},
        ]
        out = _merge_plan_once(steps, "move the marker")
        assert out[0]["target_related_object"] == "bowl"
        assert out[0]["target_location"] == "next to bowl"


class TestInstructionReplacesStepText:
    """The merged .step uses the full user instruction, not step[0].step."""

    def test_instruction_overrides_step_text(self):
        steps = [
            {"step": "decomposed step 1", "manipulating_object": "x", "target_related_object": "", "target_location": ""},
            {"step": "decomposed step 2", "manipulating_object": "x", "target_related_object": "y", "target_location": "on y"},
        ]
        instruction = "Original user instruction that mentions x and y"
        out = _merge_plan_once(steps, instruction)
        assert out[0]["step"] == instruction
        assert out[0]["step"] != "decomposed step 1"
