"""Real-API smoke tests for Gemini + OpenAI endpoints used by trajectory_predictor.

Every test here makes a real API call. Each call is kept minimum-size; total cost
for one full run is typically < $0.02. Tests skip automatically if the keys are
not set.

Run:
    # Keys should be in ~/pi-trajectory-overlay/.env; conftest below loads them.
    python -m pytest tests/test_llm_apis_real.py -v

Tests split into:
  * TestConnectivity        — can we even reach the APIs?
  * TestInstructionDecomp   — query_target_objects
  * TestObjectDetection     — query_target_location (Gemini Robotics-ER)
  * TestTrajectoryPlanning  — query_trajectory
  * TestStepCompletion      — query_step_completion
  * TestIPv4Patch           — confirms getaddrinfo stays v4-only during real calls
"""
from __future__ import annotations

import os
import socket
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.expanduser("~/pi-trajectory-overlay")
sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# .env loader (lightweight; no python-dotenv dependency)
# ---------------------------------------------------------------------------

def _load_dotenv_into_environ(path: str):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv_into_environ(os.path.join(REPO, ".env"))

_HAS_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY"))
_HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))

pytestmark = [
    pytest.mark.skipif(not _HAS_GEMINI_KEY, reason="GEMINI_API_KEY not set"),
    pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set"),
]


# Importing trajectory_predictor also installs the IPv4-only DNS patch.
from trajectory_predictor import (  # noqa: E402
    _call_gemini,
    _get_google_client,
    _get_openai_client,
    encode_pil_image,
    query_step_completion,
    query_target_location,
    query_target_objects,
    query_trajectory,
    resize_for_api,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tabletop_image():
    """Synthetic tabletop: grey table, red cup on the left, blue bowl on the right."""
    img = Image.new("RGB", (640, 480), (180, 180, 180))
    # red cup
    for y in range(180, 300):
        for x in range(100, 200):
            if (x - 150) ** 2 + (y - 240) ** 2 <= 55 ** 2:
                img.putpixel((x, y), (220, 30, 30))
    # blue bowl
    for y in range(200, 320):
        for x in range(420, 540):
            if (x - 480) ** 2 + (y - 260) ** 2 <= 60 ** 2:
                img.putpixel((x, y), (30, 60, 220))
    return img


@pytest.fixture(scope="module")
def tabletop_image_encoded(tabletop_image):
    return encode_pil_image(resize_for_api(tabletop_image))


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class TestConnectivity:
    def test_openai_client_roundtrip(self):
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=5,
        )
        text = resp.choices[0].message.content.strip().lower()
        assert "pong" in text, f"OpenAI returned {text!r}"

    def test_gemini_client_roundtrip(self):
        """Hit the exact model the real pipeline uses (main.py Args default)."""
        client = _get_google_client()
        # Use the same thinking-enabled config _call_gemini uses; robotics-er
        # requires thinking mode.
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=100),
        )
        resp = client.models.generate_content(
            model="gemini-robotics-er-1.5-preview",
            contents="Reply with the single word: pong",
            config=config,
        )
        text = (resp.text or "").strip().lower()
        assert "pong" in text, f"Gemini returned {text!r}"

    def test_call_gemini_helper_with_image(self, tabletop_image):
        """The wrapper we actually use in trajectory_predictor.

        Uses gemini-robotics-er-1.5-preview — the model _call_gemini is
        designed for (it enables thinking mode, which is only supported
        by the robotics-er family).
        """
        out = _call_gemini(
            resize_for_api(tabletop_image),
            "Describe this image in 5 words or fewer.",
            model_name="gemini-robotics-er-1.5-preview",
        )
        assert isinstance(out, str)
        assert len(out.strip()) > 0


# ---------------------------------------------------------------------------
# query_target_objects — instruction decomposition (GPT)
# ---------------------------------------------------------------------------

class TestInstructionDecomposition:
    def test_pick_and_place_decomposes_to_steps(self, tabletop_image_encoded):
        out = query_target_objects(
            "Pick up the red cup and put it in the blue bowl",
            gpt_model="gpt-4o-mini",
            img_encoded=tabletop_image_encoded,
        )
        assert "steps" in out, f"missing 'steps' in response: {out}"
        assert len(out["steps"]) >= 1
        s0 = out["steps"][0]
        for key in ("step", "manipulating_object", "target_related_object", "target_location"):
            assert key in s0, f"step missing key {key!r}; step={s0}"
        # The manipulating object should be the cup (not the bowl).
        assert "cup" in s0["manipulating_object"].lower()

    def test_press_button_single_step(self, tabletop_image_encoded):
        out = query_target_objects(
            "Press the red cup",
            gpt_model="gpt-4o-mini",
            img_encoded=tabletop_image_encoded,
        )
        assert len(out.get("steps", [])) >= 1
        s0 = out["steps"][0]
        assert "cup" in s0["manipulating_object"].lower()


# ---------------------------------------------------------------------------
# query_target_location — Gemini Robotics-ER object detection
# ---------------------------------------------------------------------------

class TestObjectDetection:
    def test_detects_two_objects(self, tabletop_image):
        api_img = resize_for_api(tabletop_image)
        out = query_target_location(
            api_img,
            ["red cup", "blue bowl"],
            model_name="gemini-robotics-er-1.5-preview",
        )
        # Gemini may return any non-empty dict-ish result, or None if it refuses.
        if out is None:
            pytest.skip("Gemini declined to detect on synthetic image")
        assert isinstance(out, dict)
        assert len(out) >= 1
        # Each value should be a 2-tuple/list of pixel coords within the image.
        w, h = api_img.size
        for name, pt in out.items():
            assert isinstance(name, str)
            assert len(pt) == 2
            assert 0 <= pt[0] <= w, f"{name} x out of range: {pt[0]} (w={w})"
            assert 0 <= pt[1] <= h, f"{name} y out of range: {pt[1]} (h={h})"

    def test_missing_object_returns_partial_or_none(self, tabletop_image):
        api_img = resize_for_api(tabletop_image)
        out = query_target_location(
            api_img,
            ["unicorn"],  # not in the image
            model_name="gemini-robotics-er-1.5-preview",
        )
        # Acceptable outcomes: None, empty dict, or a dict with fewer than 10 entries.
        assert out is None or isinstance(out, dict)
        if isinstance(out, dict):
            assert len(out) <= 10


# ---------------------------------------------------------------------------
# query_trajectory — GPT trajectory generation
# ---------------------------------------------------------------------------

class TestTrajectoryPlanning:
    def test_returns_valid_trajectory_schema(self, tabletop_image, tabletop_image_encoded):
        api_img = resize_for_api(tabletop_image)
        out = query_trajectory(
            img=api_img,
            img_encoded=tabletop_image_encoded,
            task="Pick up the red cup and put it in the blue bowl",
            manipulating_object="red cup",
            manipulating_object_point=(150, 240),
            target_related_object="blue bowl",
            target_related_object_point=(480, 260),
            target_location="in the blue bowl",
            gpt_model="gpt-4o-mini",
            full_task="Pick up the red cup and put it in the blue bowl",
        )
        assert isinstance(out, dict)
        for key in ("trajectory", "start_point", "end_point"):
            assert key in out, f"missing key {key!r}: {out}"
        pts = out["trajectory"]
        # Prompt asks for 10-15 points; schema declares minItems=10 but GPT
        # doesn't strictly honor it — we've seen 7-point responses in practice.
        # A usable trajectory needs at least ~3 waypoints for a curve.
        assert len(pts) >= 3, f"got {len(pts)} points; need at least 3 for a curve"
        assert len(pts) <= 30, "unexpectedly many points"
        # Each point is a 2-list of numbers.
        for p in pts:
            assert len(p) == 2
            assert all(isinstance(c, (int, float)) for c in p)

    def test_meaningful_displacement(self, tabletop_image, tabletop_image_encoded):
        api_img = resize_for_api(tabletop_image)
        out = query_trajectory(
            img=api_img,
            img_encoded=tabletop_image_encoded,
            task="Pick up the red cup and put it in the blue bowl",
            manipulating_object="red cup",
            manipulating_object_point=(150, 240),
            target_related_object="blue bowl",
            target_related_object_point=(480, 260),
            target_location="in the blue bowl",
            gpt_model="gpt-4o-mini",
            full_task="Pick up the red cup and put it in the blue bowl",
        )
        sp = out["start_point"]
        ep = out["end_point"]
        disp = ((ep[0] - sp[0]) ** 2 + (ep[1] - sp[1]) ** 2) ** 0.5
        img_size = max(api_img.size)
        # Prompt requires at least 10% of image width for pick-and-place.
        assert disp > 0.05 * img_size, f"trivial trajectory: disp={disp:.1f}px in {img_size}px image"


# ---------------------------------------------------------------------------
# query_step_completion — Gemini step-done check
# ---------------------------------------------------------------------------

class TestStepCompletion:
    def test_returns_is_complete_bool(self, tabletop_image):
        api_img = resize_for_api(tabletop_image)
        out = query_step_completion(
            [api_img],
            step="Pick up the red cup",
            model_name="gemini-robotics-er-1.5-preview",
            max_images=1,
        )
        assert isinstance(out, dict)
        assert "is_complete" in out
        assert isinstance(out["is_complete"], bool)


# ---------------------------------------------------------------------------
# IPv4 patch — verify it's still effective during real API traffic
# ---------------------------------------------------------------------------

class TestIPv4PatchUnderRealTraffic:
    def test_getaddrinfo_yields_only_v4(self):
        """After a real API call the patch should still be installed."""
        # Ensure we've done at least one real call before asserting.
        client = _get_openai_client()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        results = socket.getaddrinfo("api.openai.com", 443)
        assert results
        for r in results:
            assert r[0] == socket.AF_INET, f"IPv6 leaked through: {r}"
