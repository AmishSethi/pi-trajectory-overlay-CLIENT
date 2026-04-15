"""Tests for main.py helper functions and PlanningState."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PIL import Image

# We can't import the full main module without droid, so test the pieces we can
from traj_vis_utils import TraceOverlayConfig, add_trace_overlay


# ---------------------------------------------------------------------------
# _draw_trajectory_on_image (replicate logic from main.py)
# ---------------------------------------------------------------------------

def _draw_trajectory_on_image(image_rgb, trajectory_points, current_index=0, config=None):
    """Replicate the function from main.py for testing."""
    if len(trajectory_points) < 2:
        return image_rgb
    pts = [tuple(p) for p in trajectory_points]
    idx = min(current_index, len(pts) - 1)
    result = add_trace_overlay(image_rgb, pts, current_index=idx, config=config,
                               num_interpolated=0)
    return np.array(result.convert("RGB"))


class TestDrawTrajectoryOnImage:
    """Test the trajectory drawing function used in the inference loop."""

    def test_draws_on_rgb_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = [[100, 200], [300, 200], [500, 200]]
        result = _draw_trajectory_on_image(img, pts)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8
        assert result.max() > 0

    def test_single_point_returns_unchanged(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        pts = [[100, 100]]
        result = _draw_trajectory_on_image(img, pts)
        assert result.max() == 0

    def test_empty_points_returns_unchanged(self):
        img = np.full((200, 300, 3), 128, dtype=np.uint8)
        pts = []
        result = _draw_trajectory_on_image(img, pts)
        np.testing.assert_array_equal(result, img)

    def test_default_config_matches_training(self):
        """Verify the default config produces red-to-pink gradient + yellow dot."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        pts = [[50, 112], [112, 50], [174, 112]]
        result = _draw_trajectory_on_image(img, pts, current_index=1)
        # Should have red/pink gradient pixels
        red_ish = (result[:, :, 0] > 200) & (result[:, :, 1] < 120)
        assert red_ish.any(), "Default config should produce red-pink trajectory"
        # Should have yellow dot at current_index
        yellow = (result[:, :, 0] > 200) & (result[:, :, 1] > 200) & (result[:, :, 2] < 50)
        assert yellow.any(), "Default config should produce yellow dot"

    def test_custom_config(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        pts = [[50, 112], [174, 112]]
        cfg = TraceOverlayConfig(
            future_color=(0, 255, 0),
            future_thickness=3,
        )
        result = _draw_trajectory_on_image(img, pts, config=cfg)
        green = (result[:, :, 1] > 200) & (result[:, :, 0] < 50) & (result[:, :, 2] < 50)
        assert green.any(), "Custom config should produce green trajectory"

    def test_at_model_input_resolution(self):
        """Test at exact 224x224 model input resolution."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pts = [[30, 112], [60, 80], [112, 50], [164, 80], [194, 112]]
        result = _draw_trajectory_on_image(img, pts)
        assert result.shape == (224, 224, 3)
        # Result should differ from input (trajectory was drawn)
        assert not np.array_equal(result, img)


# ---------------------------------------------------------------------------
# PlanningState (replicate from main.py)
# ---------------------------------------------------------------------------

import dataclasses

@dataclasses.dataclass
class PlanningState:
    steps: list = dataclasses.field(default_factory=list)
    step_idx: int = 0
    pred_traj: dict | None = None
    current_end_point: list | None = None
    img_history: list = dataclasses.field(default_factory=list)
    plan_count: int = 0
    steps_since_last_plan: int = 0

    def reset_step(self):
        self.pred_traj = None
        self.current_end_point = None
        self.img_history = []
        self.steps_since_last_plan = 0

    @property
    def current_step(self):
        if self.step_idx < len(self.steps):
            return self.steps[self.step_idx]
        return None

    @property
    def all_steps_done(self):
        return self.step_idx >= len(self.steps)


class TestPlanningState:
    def test_initial_state(self):
        ps = PlanningState()
        assert ps.steps == []
        assert ps.step_idx == 0
        assert ps.pred_traj is None
        assert ps.all_steps_done
        assert ps.current_step is None

    def test_with_steps(self):
        ps = PlanningState(steps=[
            {"step": "Pick up cup", "manipulating_object": "cup", "target_location": "shelf", "target_related_object": "shelf"},
            {"step": "Place on shelf", "manipulating_object": "cup", "target_location": "shelf", "target_related_object": "shelf"},
        ])
        assert not ps.all_steps_done
        assert ps.current_step["step"] == "Pick up cup"

    def test_advance_step(self):
        ps = PlanningState(steps=[
            {"step": "Step 1"},
            {"step": "Step 2"},
        ])
        ps.step_idx = 1
        assert ps.current_step["step"] == "Step 2"
        ps.step_idx = 2
        assert ps.all_steps_done
        assert ps.current_step is None

    def test_reset_step(self):
        ps = PlanningState()
        ps.pred_traj = {"trajectory": [[1, 2], [3, 4]]}
        ps.current_end_point = [3, 4]
        ps.img_history = [Image.new("RGB", (100, 100))]
        ps.steps_since_last_plan = 15

        ps.reset_step()
        assert ps.pred_traj is None
        assert ps.current_end_point is None
        assert ps.img_history == []
        assert ps.steps_since_last_plan == 0

    def test_plan_count_tracking(self):
        ps = PlanningState(steps=[{"step": "test"}])
        assert ps.plan_count == 0
        ps.plan_count += 1
        assert ps.plan_count == 1


# ---------------------------------------------------------------------------
# End-to-end annotation pipeline simulation
# ---------------------------------------------------------------------------

class TestAnnotationPipeline:
    """Simulate the full annotation pipeline: raw image → annotated → resize → model input."""

    def test_full_pipeline(self):
        """Simulate: camera image → draw trajectory → resize_with_pad → verify."""
        # 1. Simulate a camera image (typical ZED resolution)
        camera_image = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)

        # 2. Define a trajectory in camera pixel coordinates
        trajectory_points = [
            [200, 360],  # start
            [400, 300],
            [600, 360],
            [800, 300],
            [1000, 360],  # end
        ]

        # 3. Draw trajectory with training defaults
        annotated = _draw_trajectory_on_image(camera_image, trajectory_points)
        assert annotated.shape == (720, 1280, 3)

        # 4. Resize to 224x224 (simulating resize_with_pad)
        pil_annotated = Image.fromarray(annotated)
        resized = pil_annotated.resize((224, 224), Image.Resampling.LANCZOS)
        model_input = np.array(resized)

        assert model_input.shape == (224, 224, 3)
        assert model_input.dtype == np.uint8

        # 5. Verify trajectory is visible in resized image
        # Check for red/pink-ish pixels from the gradient line
        red_pink = (
            (model_input[:, :, 0] > 150) &
            (model_input[:, :, 1] < 130)
        )
        assert red_pink.any(), "Trajectory should be visible after resize"

    def test_pipeline_preserves_image_quality(self):
        """Annotating shouldn't destroy the image."""
        camera_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        trajectory_points = [[100, 240], [320, 100], [540, 240]]

        annotated = _draw_trajectory_on_image(camera_image, trajectory_points)

        # Most pixels should be unchanged (trajectory is thin)
        diff = np.abs(annotated.astype(int) - camera_image.astype(int))
        changed_pixels = (diff.sum(axis=2) > 0).sum()
        total_pixels = camera_image.shape[0] * camera_image.shape[1]

        # With thickness=1, less than 5% of pixels should change
        assert changed_pixels / total_pixels < 0.05

    def test_side_by_side_comparison(self):
        """Create a side-by-side image for visual verification."""
        img = np.full((224, 224, 3), 80, dtype=np.uint8)
        pts = [[30, 112], [112, 40], [194, 112]]

        annotated = _draw_trajectory_on_image(img, pts)

        # Create side-by-side
        side_by_side = np.concatenate([img, annotated], axis=1)
        assert side_by_side.shape == (224, 448, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
