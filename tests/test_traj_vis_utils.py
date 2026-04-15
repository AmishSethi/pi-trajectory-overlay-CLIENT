"""Comprehensive tests for traj_vis_utils.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PIL import Image

from traj_vis_utils import (
    TraceOverlayConfig,
    add_trace_overlay,
    _lerp_color,
    _annotate_frame,
    _arrow_indices,
    _draw_polyline_with_outline,
    _draw_circle_with_outline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image_rgb():
    """200x300 black RGB image as numpy array."""
    return np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.fixture
def blank_pil_image():
    """200x300 black PIL image."""
    return Image.new("RGB", (300, 200), (0, 0, 0))


@pytest.fixture
def simple_points():
    """Simple 3-point trajectory."""
    return [(50, 100), (150, 50), (250, 100)]


@pytest.fixture
def long_trajectory():
    """20-point winding trajectory."""
    pts = []
    for i in range(20):
        x = 20 + i * 13
        y = 100 + 40 * np.sin(i * 0.5)
        pts.append((float(x), float(y)))
    return pts


@pytest.fixture
def two_points():
    """Minimal 2-point trajectory."""
    return [(50, 100), (250, 100)]


# ---------------------------------------------------------------------------
# TraceOverlayConfig defaults match training
# ---------------------------------------------------------------------------

class TestTraceOverlayConfigDefaults:
    """Verify the defaults exactly match what was used during training.

    Source of truth: build_dataset.py in branyang02/temp_drawing/droid_trajectory
    """

    def test_future_color_is_red(self):
        cfg = TraceOverlayConfig()
        assert cfg.future_color == (255, 0, 0), "Training used red (255, 0, 0)"

    def test_future_color_end_is_pink(self):
        cfg = TraceOverlayConfig()
        assert cfg.future_color_end == (255, 105, 180), "Training used pink gradient end"

    def test_future_thickness_is_3(self):
        cfg = TraceOverlayConfig()
        assert cfg.future_thickness == 3, "Training used thickness=3"

    def test_future_outline_thickness_is_5(self):
        cfg = TraceOverlayConfig()
        assert cfg.future_outline_thickness == 5, "Training used black outline thickness=5"

    def test_future_outline_color_is_black(self):
        cfg = TraceOverlayConfig()
        assert cfg.future_outline_color == (0, 0, 0)

    def test_current_dot_radius_is_5(self):
        cfg = TraceOverlayConfig()
        assert cfg.current_dot_radius == 5, "Training used yellow dot radius=5"

    def test_current_dot_color_is_yellow(self):
        cfg = TraceOverlayConfig()
        assert cfg.current_dot_color == (255, 255, 0), "Training used yellow dot"

    def test_current_dot_has_black_outline(self):
        cfg = TraceOverlayConfig()
        assert cfg.current_dot_outline_color == (0, 0, 0)
        assert cfg.current_dot_outline_thickness == 2

    def test_no_arrows(self):
        cfg = TraceOverlayConfig()
        assert cfg.arrow_count == 0, "Training used no arrows"

    def test_no_dashes(self):
        cfg = TraceOverlayConfig()
        assert cfg.dashed_future is False, "Training used solid lines"

    def test_no_past(self):
        cfg = TraceOverlayConfig()
        assert cfg.show_past is False, "Training showed no past trajectory"

    def test_horizon_zero_means_all(self):
        cfg = TraceOverlayConfig()
        assert cfg.horizon == 0, "horizon=0 should draw entire future"


# ---------------------------------------------------------------------------
# add_trace_overlay: input types
# ---------------------------------------------------------------------------

class TestAddTraceOverlayInputTypes:
    """Test that add_trace_overlay accepts various input types."""

    def test_numpy_input(self, blank_image_rgb, simple_points):
        result = add_trace_overlay(blank_image_rgb, simple_points)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_pil_input(self, blank_pil_image, simple_points):
        result = add_trace_overlay(blank_pil_image, simple_points)
        assert isinstance(result, Image.Image)

    def test_pil_input_not_mutated(self, blank_pil_image, simple_points):
        original_data = np.array(blank_pil_image).copy()
        add_trace_overlay(blank_pil_image, simple_points)
        np.testing.assert_array_equal(np.array(blank_pil_image), original_data)

    def test_file_path_input(self, tmp_path, simple_points):
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (300, 200), (128, 128, 128)).save(img_path)
        result = add_trace_overlay(img_path, simple_points)
        assert isinstance(result, Image.Image)

    def test_rgba_input(self, simple_points):
        rgba = np.zeros((200, 300, 4), dtype=np.uint8)
        result = add_trace_overlay(rgba, simple_points)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_invalid_input_raises(self, simple_points):
        with pytest.raises(TypeError):
            add_trace_overlay(12345, simple_points)


# ---------------------------------------------------------------------------
# add_trace_overlay: output properties
# ---------------------------------------------------------------------------

class TestAddTraceOverlayOutput:
    """Test output properties of add_trace_overlay."""

    def test_output_same_size_as_input(self, blank_image_rgb, simple_points):
        result = add_trace_overlay(blank_image_rgb, simple_points)
        assert result.size == (300, 200)  # PIL size is (W, H)

    def test_draws_something_on_black_image(self, blank_image_rgb, simple_points):
        result = add_trace_overlay(blank_image_rgb, simple_points)
        result_arr = np.array(result)
        assert result_arr.max() > 0, "Should have drawn something on the black image"

    def test_red_pink_gradient_present(self, blank_image_rgb, simple_points):
        """Default config is red-to-pink gradient with black outline."""
        result = add_trace_overlay(blank_image_rgb, simple_points)
        arr = np.array(result)
        # Look for red-ish pixels (R > 200, G < 120, B < 200)
        red_ish = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 120)
        assert red_ish.any(), "Should contain red/pink pixels with default config"

    def test_yellow_dot_present(self, blank_image_rgb, simple_points):
        """Default config draws a yellow dot at current_index."""
        result = add_trace_overlay(blank_image_rgb, simple_points, current_index=1)
        arr = np.array(result)
        # Yellow: R > 200, G > 200, B < 50
        yellow = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 50)
        assert yellow.any(), "Should contain yellow dot pixels"

    def test_black_outline_present(self, blank_image_rgb, simple_points):
        """Default config has black outline around the trajectory line."""
        # Use a non-black background to detect the outline
        bg = np.full((200, 300, 3), 128, dtype=np.uint8)
        result = add_trace_overlay(bg, simple_points)
        arr = np.array(result)
        # Black pixels that weren't in the original (outline)
        black = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)
        assert black.any(), "Should contain black outline pixels"

    def test_colored_image_preserves_background(self, simple_points):
        """Background pixels should be preserved."""
        bg = np.full((200, 300, 3), 100, dtype=np.uint8)
        result = add_trace_overlay(bg, simple_points)
        arr = np.array(result)
        # Pixels far from the trajectory should still be ~100
        corner = arr[0, 0]
        np.testing.assert_array_equal(corner, [100, 100, 100])


# ---------------------------------------------------------------------------
# add_trace_overlay: spline interpolation
# ---------------------------------------------------------------------------

class TestSplineInterpolation:
    """Test spline smoothing and interpolation."""

    def test_num_interpolated_zero_uses_raw_points(self, blank_image_rgb, simple_points):
        """With num_interpolated=0, should use raw points directly."""
        result = add_trace_overlay(blank_image_rgb, simple_points, num_interpolated=0)
        assert isinstance(result, Image.Image)

    def test_two_points_linear_interpolation(self, blank_image_rgb, two_points):
        """Two points should use linear interpolation (not spline)."""
        result = add_trace_overlay(blank_image_rgb, two_points)
        assert isinstance(result, Image.Image)

    def test_high_num_interpolated(self, blank_image_rgb, simple_points):
        """High interpolation count should produce smooth result."""
        result = add_trace_overlay(blank_image_rgb, simple_points, num_interpolated=1000)
        assert isinstance(result, Image.Image)

    def test_smoothing_parameter(self, blank_image_rgb, long_trajectory):
        """Non-zero smoothing with interpolation should produce a smoother curve."""
        # Must enable interpolation for smoothing to take effect
        result_smooth = add_trace_overlay(blank_image_rgb, long_trajectory,
                                          num_interpolated=300, smoothing=10.0)
        result_exact = add_trace_overlay(blank_image_rgb, long_trajectory,
                                         num_interpolated=300, smoothing=0.0)
        assert isinstance(result_smooth, Image.Image)
        assert isinstance(result_exact, Image.Image)
        arr_smooth = np.array(result_smooth)
        arr_exact = np.array(result_exact)
        assert not np.array_equal(arr_smooth, arr_exact)


# ---------------------------------------------------------------------------
# add_trace_overlay: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_point_returns_unchanged(self, blank_image_rgb):
        """Single point should return image unchanged."""
        result = add_trace_overlay(blank_image_rgb, [(100, 100)])
        arr = np.array(result)
        assert arr.max() == 0, "Single point should not draw anything"

    def test_empty_points_raises(self, blank_image_rgb):
        with pytest.raises(ValueError):
            add_trace_overlay(blank_image_rgb, [])

    def test_invalid_points_shape_raises(self, blank_image_rgb):
        with pytest.raises(ValueError):
            add_trace_overlay(blank_image_rgb, [(1, 2, 3), (4, 5, 6)])

    def test_duplicate_points(self, blank_image_rgb):
        """Duplicate points should not crash."""
        result = add_trace_overlay(blank_image_rgb, [(100, 100), (100, 100), (100, 100)])
        assert isinstance(result, Image.Image)

    def test_very_close_points(self, blank_image_rgb):
        """Very close points should not crash."""
        pts = [(100.0, 100.0), (100.001, 100.001), (100.002, 100.002)]
        result = add_trace_overlay(blank_image_rgb, pts)
        assert isinstance(result, Image.Image)

    def test_points_outside_image(self, blank_image_rgb):
        """Points outside image bounds should not crash."""
        pts = [(-50, -50), (500, 500)]
        result = add_trace_overlay(blank_image_rgb, pts)
        assert isinstance(result, Image.Image)

    def test_large_trajectory(self, blank_image_rgb):
        """100 points should work fine."""
        pts = [(float(i * 3), float(100 + 50 * np.sin(i * 0.1))) for i in range(100)]
        result = add_trace_overlay(blank_image_rgb, pts)
        assert isinstance(result, Image.Image)

    def test_current_index_middle(self, blank_image_rgb, simple_points):
        """current_index in middle should work."""
        result = add_trace_overlay(blank_image_rgb, simple_points, current_index=1)
        assert isinstance(result, Image.Image)

    def test_current_index_end(self, blank_image_rgb, simple_points):
        """current_index at end should work."""
        result = add_trace_overlay(blank_image_rgb, simple_points, current_index=2)
        assert isinstance(result, Image.Image)

    def test_current_index_clamped(self, blank_image_rgb, simple_points):
        """current_index beyond range should be clamped."""
        result = add_trace_overlay(blank_image_rgb, simple_points, current_index=999)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Custom configs
# ---------------------------------------------------------------------------

class TestCustomConfigs:
    """Test various non-default TraceOverlayConfig options."""

    def test_color_gradient(self, blank_image_rgb, simple_points):
        cfg = TraceOverlayConfig(
            future_color=(255, 0, 0),
            future_color_end=(0, 0, 255),
            future_thickness=3,
        )
        result = add_trace_overlay(blank_image_rgb, simple_points, config=cfg)
        arr = np.array(result)
        assert arr.max() > 0

    def test_outline(self, blank_image_rgb, simple_points):
        cfg = TraceOverlayConfig(
            future_color=(255, 0, 255),
            future_thickness=2,
            future_outline_thickness=5,
            future_outline_color=(255, 255, 0),
        )
        result = add_trace_overlay(blank_image_rgb, simple_points, config=cfg)
        arr = np.array(result)
        # Should have yellow outline pixels
        yellow_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 50)
        assert yellow_mask.any()

    def test_dashed_line(self, blank_image_rgb, simple_points):
        cfg = TraceOverlayConfig(
            future_color=(255, 0, 255),
            future_thickness=2,
            dashed_future=True,
            dash_len=10,
            gap_len=10,
        )
        result = add_trace_overlay(blank_image_rgb, simple_points, config=cfg)
        assert isinstance(result, Image.Image)

    def test_arrows(self, blank_image_rgb, simple_points):
        cfg = TraceOverlayConfig(
            future_color=(255, 0, 255),
            future_thickness=2,
            arrow_count=1,
            arrow_size=14,
            arrow_thickness=2,
            arrow_color=(0, 255, 0),
        )
        result = add_trace_overlay(blank_image_rgb, simple_points, config=cfg)
        arr = np.array(result)
        green_mask = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 50) & (arr[:, :, 2] < 50)
        assert green_mask.any(), "Should have green arrow pixels"

    def test_current_dot(self, blank_image_rgb, simple_points):
        cfg = TraceOverlayConfig(
            current_dot_radius=8,
            current_dot_color=(255, 255, 0),
        )
        result = add_trace_overlay(blank_image_rgb, simple_points, current_index=1, config=cfg)
        arr = np.array(result)
        yellow_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 50)
        assert yellow_mask.any(), "Should have yellow dot pixels"

    def test_past_trajectory(self, blank_image_rgb, long_trajectory):
        cfg = TraceOverlayConfig(
            show_past=True,
            past_color=(180, 180, 180),
            past_thickness=2,
            future_thickness=3,
        )
        result = add_trace_overlay(blank_image_rgb, long_trajectory, current_index=10, config=cfg)
        arr = np.array(result)
        gray_mask = (
            (arr[:, :, 0] > 150) & (arr[:, :, 0] < 200) &
            (arr[:, :, 1] > 150) & (arr[:, :, 1] < 200) &
            (arr[:, :, 2] > 150) & (arr[:, :, 2] < 200)
        )
        assert gray_mask.any(), "Should have gray past-trajectory pixels"

    def test_tick_marks(self, blank_image_rgb, long_trajectory):
        cfg = TraceOverlayConfig(
            tick_marks=True,
            tick_every=3,
            tick_radius=3,
            tick_color=(255, 255, 255),
        )
        result = add_trace_overlay(blank_image_rgb, long_trajectory, config=cfg)
        arr = np.array(result)
        white_mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        assert white_mask.any(), "Should have white tick mark pixels"

    def test_alpha_blending(self, simple_points):
        bg = np.full((200, 300, 3), 128, dtype=np.uint8)
        cfg = TraceOverlayConfig(
            future_color=(255, 0, 255),
            future_thickness=5,
            use_alpha=True,
            alpha=0.5,
        )
        result = add_trace_overlay(bg, simple_points, config=cfg)
        arr = np.array(result)
        # With alpha=0.5 on gray bg, magenta pixels should be blended
        # They shouldn't be pure magenta (255, 0, 255)
        # but something like (191, 64, 191)
        assert isinstance(result, Image.Image)

    def test_horizon_limits_future(self, blank_image_rgb, long_trajectory):
        cfg_full = TraceOverlayConfig(horizon=0, future_thickness=3)
        cfg_short = TraceOverlayConfig(horizon=3, future_thickness=3)
        result_full = np.array(add_trace_overlay(blank_image_rgb, long_trajectory, config=cfg_full))
        result_short = np.array(add_trace_overlay(blank_image_rgb, long_trajectory, config=cfg_short))
        # Short horizon should have fewer colored pixels
        full_colored = (result_full.sum(axis=2) > 0).sum()
        short_colored = (result_short.sum(axis=2) > 0).sum()
        assert short_colored < full_colored


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestLerpColor:
    def test_t0_returns_c0(self):
        assert _lerp_color((255, 0, 0), (0, 255, 0), 0.0) == (255, 0, 0)

    def test_t1_returns_c1(self):
        assert _lerp_color((255, 0, 0), (0, 255, 0), 1.0) == (0, 255, 0)

    def test_t_half_returns_midpoint(self):
        assert _lerp_color((0, 0, 0), (200, 100, 50), 0.5) == (100, 50, 25)

    def test_same_colors(self):
        assert _lerp_color((128, 128, 128), (128, 128, 128), 0.7) == (128, 128, 128)


class TestArrowIndices:
    def test_zero_arrows(self):
        cfg = TraceOverlayConfig(arrow_count=0)
        assert _arrow_indices(10, cfg) == []

    def test_end_only_arrow(self):
        cfg = TraceOverlayConfig(arrow_count=1, arrow_mode="end_only")
        assert _arrow_indices(10, cfg) == [8]

    def test_multiple_arrows(self):
        cfg = TraceOverlayConfig(arrow_count=3, arrow_mode="multiple")
        indices = _arrow_indices(10, cfg)
        assert len(indices) == 3
        # Should be roughly evenly spaced
        assert all(0 <= i < 9 for i in indices)

    def test_too_few_points(self):
        cfg = TraceOverlayConfig(arrow_count=5)
        assert _arrow_indices(1, cfg) == []


# ---------------------------------------------------------------------------
# 224x224 model input simulation
# ---------------------------------------------------------------------------

class TestModelInputSize:
    """Test trajectory overlay at the exact size the model receives (224x224)."""

    def test_overlay_at_224x224(self, simple_points):
        """Draw on a 224x224 image like the model sees."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Scale points to 224x224
        scaled_pts = [(p[0] * 224 / 300, p[1] * 224 / 200) for p in simple_points]
        result = add_trace_overlay(img, scaled_pts)
        assert result.size == (224, 224)
        arr = np.array(result)
        assert arr.max() > 0

    def test_overlay_then_resize_vs_resize_then_overlay(self, simple_points):
        """Compare drawing on full-res then resizing vs resizing then drawing."""
        # This tests that the pipeline draws on full-res BEFORE resize_with_pad
        full_res = np.zeros((480, 640, 3), dtype=np.uint8)
        result_full = add_trace_overlay(full_res, simple_points)
        assert result_full.size == (640, 480)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
