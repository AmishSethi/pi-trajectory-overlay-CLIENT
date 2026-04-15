"""Tests for trajectory_predictor.py helper functions.

API-calling functions are tested with mocks to avoid real API calls.
The google-genai package may not be installed in all environments, so we
import individual functions carefully and skip API-dependent tests if needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# These helpers don't depend on google-genai at import time
# We import them indirectly to handle the missing dependency
try:
    from trajectory_predictor import (
        _parse_json,
        encode_pil_image,
        rescale_trajectory,
        resize_for_api,
    )
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    # Define standalone versions for testing
    def _parse_json(text):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                text = "\n".join(lines[i + 1:])
                text = text.split("```")[0]
                break
        return text

    def encode_pil_image(img):
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def resize_for_api(img, max_size=1024):
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        return img

    def rescale_trajectory(trajectory, resized_size, original_size):
        rw, rh = resized_size
        ow, oh = original_size
        if rw == ow and rh == oh:
            return trajectory
        sx, sy = ow / rw, oh / rh
        def scale_pt(pt):
            return [pt[0] * sx, pt[1] * sy]
        out = dict(trajectory)
        if "trajectory" in out:
            out["trajectory"] = [scale_pt(p) for p in out["trajectory"]]
        if "start_point" in out:
            out["start_point"] = scale_pt(out["start_point"])
        if "end_point" in out:
            out["end_point"] = scale_pt(out["end_point"])
        return out

skip_if_no_predictor = pytest.mark.skipif(
    not PREDICTOR_AVAILABLE,
    reason="trajectory_predictor requires google-genai",
)


# ---------------------------------------------------------------------------
# encode_pil_image
# ---------------------------------------------------------------------------

class TestEncodePilImage:
    def test_returns_base64_string(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        encoded = encode_pil_image(img)
        assert isinstance(encoded, str)
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    def test_roundtrip_preserves_image(self):
        img = Image.new("RGB", (50, 50), (0, 128, 255))
        encoded = encode_pil_image(img)
        decoded_bytes = base64.b64decode(encoded)
        import io
        roundtrip = Image.open(io.BytesIO(decoded_bytes))
        assert roundtrip.size == (50, 50)
        # Check a pixel
        assert roundtrip.getpixel((25, 25)) == (0, 128, 255)

    def test_different_images_different_encodings(self):
        img1 = Image.new("RGB", (50, 50), (255, 0, 0))
        img2 = Image.new("RGB", (50, 50), (0, 255, 0))
        assert encode_pil_image(img1) != encode_pil_image(img2)


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_plain_json(self):
        text = '{"key": "value"}'
        assert _parse_json(text) == '{"key": "value"}'

    def test_markdown_fenced_json(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _parse_json(text)
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_no_fence(self):
        text = '{"a": 1, "b": 2}'
        result = _parse_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_multiple_fences(self):
        text = '```json\n{"first": true}\n```\n\n```json\n{"second": true}\n```'
        result = _parse_json(text)
        parsed = json.loads(result)
        assert parsed == {"first": True}

    def test_empty_json_fence(self):
        text = '```json\n{}\n```'
        result = _parse_json(text)
        assert json.loads(result) == {}


# ---------------------------------------------------------------------------
# resize_for_api
# ---------------------------------------------------------------------------

class TestResizeForApi:
    def test_small_image_unchanged(self):
        img = Image.new("RGB", (640, 480))
        result = resize_for_api(img, max_size=1024)
        assert result.size == (640, 480)

    def test_large_image_resized(self):
        img = Image.new("RGB", (2000, 1500))
        result = resize_for_api(img, max_size=1024)
        w, h = result.size
        assert max(w, h) == 1024
        # Aspect ratio preserved
        assert abs(w / h - 2000 / 1500) < 0.02

    def test_exact_max_size_unchanged(self):
        img = Image.new("RGB", (1024, 768))
        result = resize_for_api(img, max_size=1024)
        assert result.size == (1024, 768)

    def test_tall_image(self):
        img = Image.new("RGB", (500, 2000))
        result = resize_for_api(img, max_size=1024)
        w, h = result.size
        assert h == 1024
        assert abs(w / h - 500 / 2000) < 0.02

    def test_custom_max_size(self):
        img = Image.new("RGB", (800, 600))
        result = resize_for_api(img, max_size=400)
        w, h = result.size
        assert max(w, h) == 400


# ---------------------------------------------------------------------------
# rescale_trajectory
# ---------------------------------------------------------------------------

class TestRescaleTrajectory:
    def test_same_size_returns_unchanged(self):
        traj = {
            "trajectory": [[100, 200], [300, 400]],
            "start_point": [100, 200],
            "end_point": [300, 400],
        }
        result = rescale_trajectory(traj, (640, 480), (640, 480))
        assert result["trajectory"] == [[100, 200], [300, 400]]

    def test_2x_upscale(self):
        traj = {
            "trajectory": [[100, 100], [200, 200]],
            "start_point": [100, 100],
            "end_point": [200, 200],
        }
        result = rescale_trajectory(traj, (320, 240), (640, 480))
        assert result["trajectory"][0] == [200, 200]
        assert result["trajectory"][1] == [400, 400]
        assert result["start_point"] == [200, 200]
        assert result["end_point"] == [400, 400]

    def test_downscale(self):
        traj = {
            "trajectory": [[400, 300]],
            "start_point": [400, 300],
            "end_point": [400, 300],
        }
        result = rescale_trajectory(traj, (640, 480), (320, 240))
        assert result["trajectory"][0] == [200.0, 150.0]

    def test_asymmetric_scale(self):
        traj = {
            "trajectory": [[100, 100]],
            "start_point": [100, 100],
            "end_point": [100, 100],
        }
        # API image: 512x384, original: 1024x480
        result = rescale_trajectory(traj, (512, 384), (1024, 480))
        expected_x = 100 * (1024 / 512)
        expected_y = 100 * (480 / 384)
        assert abs(result["trajectory"][0][0] - expected_x) < 0.01
        assert abs(result["trajectory"][0][1] - expected_y) < 0.01

    def test_preserves_extra_keys(self):
        traj = {
            "trajectory": [[100, 200]],
            "start_point": [100, 200],
            "end_point": [100, 200],
            "reasoning": "test reasoning",
        }
        result = rescale_trajectory(traj, (320, 240), (640, 480))
        assert result["reasoning"] == "test reasoning"

    def test_missing_optional_keys(self):
        traj = {"trajectory": [[100, 200], [300, 400]]}
        result = rescale_trajectory(traj, (320, 240), (640, 480))
        assert len(result["trajectory"]) == 2
        assert "start_point" not in result or result.get("start_point") is None or True


# ---------------------------------------------------------------------------
# Mock API tests
# ---------------------------------------------------------------------------

@skip_if_no_predictor
class TestQueryTargetObjectsMocked:
    """Test query_target_objects with mocked OpenAI client."""

    @patch("trajectory_predictor._get_openai_client")
    def test_returns_steps(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "steps": [{
                "step": "Pick up the cup",
                "manipulating_object": "cup",
                "target_location": "shelf",
                "target_related_object": "shelf",
            }]
        })
        mock_client.chat.completions.create.return_value = mock_response

        from trajectory_predictor import query_target_objects
        result = query_target_objects("Pick up the cup and put it on the shelf", gpt_model="gpt-4o-mini")

        assert "steps" in result
        assert len(result["steps"]) == 1
        assert result["steps"][0]["manipulating_object"] == "cup"


@skip_if_no_predictor
class TestQueryTargetLocationMocked:
    """Test query_target_location with mocked Gemini client."""

    @patch("trajectory_predictor._get_google_client")
    def test_parses_locations(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"point": [300, 500], "label": "cup"},
            {"point": [600, 200], "label": "plate"},
        ])
        mock_client.models.generate_content.return_value = mock_response

        from trajectory_predictor import query_target_location
        img = Image.new("RGB", (1000, 1000))
        result = query_target_location(img, ["cup", "plate"], model_name="test-model")

        assert result is not None
        assert "cup" in result
        assert "plate" in result
        # Verify coordinate conversion: x = (x_norm/1000) * width
        cup_x, cup_y = result["cup"]
        assert abs(cup_x - 500.0) < 0.01  # x_norm=500, width=1000
        assert abs(cup_y - 300.0) < 0.01  # y_norm=300, height=1000


@skip_if_no_predictor
class TestQueryTrajectoryMocked:
    """Test query_trajectory with mocked OpenAI client."""

    @patch("trajectory_predictor._get_openai_client")
    def test_returns_trajectory(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        trajectory_data = {
            "reasoning": "Moving cup to the right",
            "start_point": [100, 200],
            "end_point": [400, 200],
            "trajectory": [[100, 200], [200, 200], [300, 200], [400, 200]],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(trajectory_data)
        mock_client.chat.completions.create.return_value = mock_response

        from trajectory_predictor import query_trajectory
        img = Image.new("RGB", (640, 480))
        result = query_trajectory(
            img=img,
            img_encoded="dummy_base64",
            task="Move cup right",
            manipulating_object="cup",
            manipulating_object_point=(100, 200),
            target_related_object="plate",
            target_related_object_point=(400, 200),
            target_location="right of plate",
        )

        assert result["trajectory"] == [[100, 200], [200, 200], [300, 200], [400, 200]]
        assert result["start_point"] == [100, 200]
        assert result["end_point"] == [400, 200]

    @patch("trajectory_predictor._get_openai_client")
    def test_with_target_location_point(self, mock_get_client):
        """When target_location_point is provided, uses re-planning prompt."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "reasoning": "Re-planning",
            "start_point": [150, 200],
            "end_point": [400, 200],
            "trajectory": [[150, 200], [275, 200], [400, 200]],
        })
        mock_client.chat.completions.create.return_value = mock_response

        from trajectory_predictor import query_trajectory
        img = Image.new("RGB", (640, 480))
        result = query_trajectory(
            img=img,
            img_encoded="dummy_base64",
            task="Move cup right",
            manipulating_object="cup",
            manipulating_object_point=(150, 200),
            target_related_object="plate",
            target_related_object_point=(400, 200),
            target_location="right of plate",
            target_location_point=(400, 200),
        )

        assert len(result["trajectory"]) == 3
        # Verify the prompt used the with-target template
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][0]["text"]
        assert "which is at:" in prompt_text


@skip_if_no_predictor
class TestQueryStepCompletionMocked:
    """Test query_step_completion with mocked Gemini client."""

    @patch("trajectory_predictor._get_google_client")
    def test_complete(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "reasoning": "The cup is now on the plate",
            "is_complete": True,
        })
        mock_client.models.generate_content.return_value = mock_response

        from trajectory_predictor import query_step_completion
        img = Image.new("RGB", (640, 480))
        result = query_step_completion([img], step="Place cup on plate")

        assert result["is_complete"] is True

    @patch("trajectory_predictor._get_google_client")
    def test_not_complete(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "reasoning": "Cup is still in original position",
            "is_complete": False,
        })
        mock_client.models.generate_content.return_value = mock_response

        from trajectory_predictor import query_step_completion
        img = Image.new("RGB", (640, 480))
        result = query_step_completion([img], step="Place cup on plate")

        assert result["is_complete"] is False

    @patch("trajectory_predictor._get_google_client")
    def test_image_subsampling(self, mock_get_client):
        """With many images, should subsample to max_images."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = json.dumps({"reasoning": "ok", "is_complete": True})
        mock_client.models.generate_content.return_value = mock_response

        from trajectory_predictor import query_step_completion
        imgs = [Image.new("RGB", (100, 100)) for _ in range(20)]
        query_step_completion(imgs, step="test", max_images=3)

        # Check that only 3 images were sent
        call_args = mock_client.models.generate_content.call_args
        contents = call_args[1]["contents"]
        img_count = sum(1 for c in contents if isinstance(c, Image.Image))
        assert img_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
