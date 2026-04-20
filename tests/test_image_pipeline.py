"""Tests for the image preprocessing pipeline.

Structure:
  * Unit tests — pure-function tests, no hardware, no network (always run).
  * Hardware tests (TestHardware) — real ZED frames, verify orientation, BGR→RGB,
    and that `_extract_observation` vertical-flips the externals but not the wrist.
  * End-to-end trajectory tests (TestHardwareTrajectoryE2E) — real Gemini + GPT
    calls to generate an actual trajectory on a live camera frame.

The hardware/E2E fixtures are module-scoped so the RobotEnv is created once and
shared across every hardware test in the file (cameras can only be opened
exclusively, so per-class fixtures would fight each other).

All hardware outputs land in /tmp/test_image_pipeline/ for visual inspection.

Run:
    python -m pytest tests/test_image_pipeline.py -v
    TEST_INSTRUCTION="Pick up the block" python -m pytest tests/test_image_pipeline.py::TestHardwareTrajectoryE2E -v
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import socket
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.expanduser("~/pi-trajectory-overlay")
sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# Lightweight .env loader (no python-dotenv dep)
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


# ---------------------------------------------------------------------------
# Module imports under test
# ---------------------------------------------------------------------------

try:
    from main import (
        Args,
        PlanningState,
        _draw_trajectory_on_image,
        _extract_observation,
        _predict_trajectory,
    )
    _MAIN_EXC = None
except Exception as e:  # pragma: no cover
    Args = None
    PlanningState = None
    _draw_trajectory_on_image = None
    _extract_observation = None
    _predict_trajectory = None
    _MAIN_EXC = f"{type(e).__name__}: {e}"

from traj_vis_utils import TraceOverlayConfig
from trajectory_predictor import (
    encode_pil_image,
    query_target_objects,
    rescale_trajectory,
    resize_for_api,
)
from openpi_client import image_tools


# ---------------------------------------------------------------------------
# Unit tests — pure functions only, always run
# ---------------------------------------------------------------------------

class TestResizeWithPad:
    def test_square_input_returns_224(self):
        img = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
        out = image_tools.resize_with_pad(img, 224, 224)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_landscape_input_letterboxes(self):
        img = np.full((720, 1280, 3), 200, dtype=np.uint8)
        out = image_tools.resize_with_pad(img, 224, 224)
        assert out.shape == (224, 224, 3)
        assert out[0, 112, :].mean() < 50, "top row should be pad"
        assert out[-1, 112, :].mean() < 50, "bottom row should be pad"
        assert out[112, 112, :].mean() > 150, "center should be content"

    def test_portrait_input_letterboxes(self):
        img = np.full((1280, 720, 3), 200, dtype=np.uint8)
        out = image_tools.resize_with_pad(img, 224, 224)
        assert out.shape == (224, 224, 3)
        assert out[112, 0, :].mean() < 50, "left col should be pad"
        assert out[112, -1, :].mean() < 50, "right col should be pad"
        assert out[112, 112, :].mean() > 150, "center should be content"


class TestResizeForApi:
    def test_small_image_unchanged(self):
        img = Image.new("RGB", (800, 600), (0, 0, 0))
        out = resize_for_api(img, max_size=1024)
        assert out.size == (800, 600)

    def test_large_image_capped_at_max(self):
        img = Image.new("RGB", (2560, 1440), (0, 0, 0))
        out = resize_for_api(img, max_size=1024)
        assert max(out.size) == 1024
        assert abs((out.size[0] / out.size[1]) - (2560 / 1440)) < 1e-3

    def test_square_image_capped(self):
        img = Image.new("RGB", (1500, 1500), (0, 0, 0))
        out = resize_for_api(img, max_size=1024)
        assert out.size == (1024, 1024)


class TestEncodePilImage:
    def test_roundtrip_preserves_pixels(self):
        img = Image.new("RGB", (64, 64), (128, 64, 192))
        img.putpixel((10, 20), (255, 0, 0))
        encoded = encode_pil_image(img)
        assert isinstance(encoded, str)
        raw = base64.b64decode(encoded)
        restored = Image.open(io.BytesIO(raw)).convert("RGB")
        assert restored.size == img.size
        assert restored.getpixel((10, 20)) == (255, 0, 0)

    def test_output_is_valid_base64(self):
        img = Image.new("RGB", (10, 10), (0, 0, 0))
        encoded = encode_pil_image(img)
        base64.b64decode(encoded, validate=True)


class TestRescaleTrajectory:
    def test_same_size_no_change(self):
        traj = {"trajectory": [[10.0, 20.0], [30.0, 40.0]], "start_point": [10.0, 20.0], "end_point": [30.0, 40.0]}
        out = rescale_trajectory(traj, (512, 384), (512, 384))
        assert out["trajectory"] == traj["trajectory"]

    def test_doubling_doubles_coordinates(self):
        traj = {"trajectory": [[10.0, 20.0], [30.0, 40.0]], "start_point": [10.0, 20.0], "end_point": [30.0, 40.0]}
        out = rescale_trajectory(traj, (512, 384), (1024, 768))
        assert out["trajectory"] == [[20.0, 40.0], [60.0, 80.0]]
        assert out["start_point"] == [20.0, 40.0]
        assert out["end_point"] == [60.0, 80.0]

    def test_aspect_ratio_change(self):
        traj = {"trajectory": [[100.0, 100.0]]}
        out = rescale_trajectory(traj, (500, 500), (1000, 200))
        assert out["trajectory"] == [[200.0, 40.0]]


@pytest.mark.skipif(_MAIN_EXC is not None, reason=f"main.py import failed: {_MAIN_EXC}")
class TestDrawTrajectoryOnImage:
    def test_no_points_returns_original(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        out = _draw_trajectory_on_image(img, [])
        np.testing.assert_array_equal(out, img)

    def test_single_point_returns_original(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        out = _draw_trajectory_on_image(img, [[10.0, 20.0]])
        np.testing.assert_array_equal(out, img)

    def test_sparse_points_produce_dense_overlay(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        pts = [[30, 100], [128, 50], [226, 200]]
        out = _draw_trajectory_on_image(img, pts, current_index=1)
        lit_red = (out[:, :, 0] > 150).sum()
        assert lit_red > 100, f"only {lit_red} red-ish pixels drawn"
        lit_yellow = ((out[:, :, 0] > 200) & (out[:, :, 1] > 200) & (out[:, :, 2] < 100)).sum()
        assert lit_yellow > 10, f"no yellow dot rendered ({lit_yellow} px)"

    def test_current_index_clamped_to_range(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        pts = [[10, 10], [60, 60], [110, 110]]
        out = _draw_trajectory_on_image(img, pts, current_index=999)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Hardware probes + gates
# ---------------------------------------------------------------------------

def _zed_cams_reachable() -> tuple[bool, str]:
    """Can we open the LEFT external + WRIST ZED cameras?"""
    try:
        import pyzed.sl as sl
    except Exception as e:
        return False, f"pyzed not importable: {e}"
    for sn in (26368109, 15512737):
        init = sl.InitParameters()
        init.set_from_serial_number(sn)
        init.depth_mode = sl.DEPTH_MODE.NONE
        cam = sl.Camera()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            cam.close()
            return False, f"SN {sn}: {status}"
        cam.close()
    return True, ""


def _on_laptop() -> bool:
    return socket.gethostname() == "franka-Alienware-x17-R2"


HARDWARE_ENABLED = (
    _MAIN_EXC is None
    and (os.environ.get("RUN_HARDWARE_TESTS") == "1" or _on_laptop())
)
if HARDWARE_ENABLED:
    _cams_ok, _cams_reason = _zed_cams_reachable()
else:
    _cams_ok, _cams_reason = False, "skipped (not on laptop / RUN_HARDWARE_TESTS not set)"

_HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_HAS_API_KEYS = _HAS_GEMINI and _HAS_OPENAI


# ---------------------------------------------------------------------------
# Module-scoped hardware fixtures — shared across hardware classes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def robot_env():
    """Single RobotEnv shared by all hardware tests in this module."""
    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    yield env
    # No explicit close API on DROID RobotEnv.


@pytest.fixture(scope="module")
def args():
    a = Args()
    a.left_camera_id = "26368109"
    a.right_camera_id = "25455306"
    a.wrist_camera_id = "15512737"
    a.external_camera = "left"
    return a


@pytest.fixture(scope="module")
def warmed_obs(robot_env, args):
    """Run the warmup loop; return the first observation with required cams ready."""
    import time
    required_ext = args.left_camera_id
    for _ in range(120):
        obs = robot_env.get_observation()
        keys = list(obs.get("image", {}).keys())
        have_ext = any(required_ext in k and "left" in k for k in keys)
        have_wrist = any(args.wrist_camera_id in k and "left" in k for k in keys)
        if have_ext and have_wrist:
            return obs
        time.sleep(0.5)
    pytest.fail(f"cameras never warmed up. Last keys: {keys}")


@pytest.fixture(scope="module")
def out_dir():
    d = "/tmp/test_image_pipeline"
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


# ---------------------------------------------------------------------------
# Hardware orientation + pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HARDWARE_ENABLED, reason="not on robot laptop")
@pytest.mark.skipif(_MAIN_EXC is not None, reason=f"main.py import failed: {_MAIN_EXC}")
@pytest.mark.skipif(not _cams_ok, reason=f"ZED cameras not available: {_cams_reason}")
class TestHardware:
    """Read real ZED frames; confirm shape, BGR→RGB channel swap, and the
    external-camera vertical flip from `_extract_observation`.

    Saves visual outputs to /tmp/test_image_pipeline/ for eyeball inspection.
    """

    def test_raw_observation_structure(self, warmed_obs):
        assert "image" in warmed_obs
        assert "robot_state" in warmed_obs
        imgs = warmed_obs["image"]
        assert isinstance(imgs, dict)
        assert len(imgs) >= 2  # at least external + wrist

    def test_captured_frame_shape_and_dtype(self, warmed_obs, args, out_dir):
        imgs = warmed_obs["image"]
        key = next(k for k in imgs if args.left_camera_id in k and "left" in k)
        frame = imgs[key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 4, f"expected BGRA 4 channels, got {frame.shape}"
        assert frame.shape[0] >= 180 and frame.shape[1] >= 320, f"tiny frame: {frame.shape}"
        # Save the RAW orientation (before _extract_observation applies the flip)
        # — this is for documentation of what comes off the ZED directly.
        Image.fromarray(frame[..., :3][..., ::-1]).save(
            os.path.join(out_dir, "01_left_raw_rgb.jpg")
        )

    def test_captured_frame_has_content(self, warmed_obs, args):
        imgs = warmed_obs["image"]
        key = next(k for k in imgs if args.left_camera_id in k and "left" in k)
        frame = imgs[key]
        std = frame[..., :3].std()
        assert std > 5.0, f"frame looks blank (std={std:.1f})"

    def test_extract_observation_shapes(self, warmed_obs, args, out_dir):
        out = _extract_observation(args, warmed_obs)
        assert out["left_image"].shape[-1] == 3
        assert out["wrist_image"].shape[-1] == 3
        assert out["left_image"].dtype == np.uint8
        assert out["joint_position"].shape == (7,)
        assert out["gripper_position"].shape == (1,)
        Image.fromarray(out["left_image"]).save(
            os.path.join(out_dir, "02_left_after_extract.jpg")
        )
        Image.fromarray(out["wrist_image"]).save(
            os.path.join(out_dir, "03_wrist_after_extract.jpg")
        )

    def test_bgr_to_rgb_conversion_correct_on_real_frame(self, warmed_obs, args):
        imgs = warmed_obs["image"]
        key = next(k for k in imgs if args.left_camera_id in k and "left" in k)
        bgra = imgs[key]
        out = _extract_observation(args, warmed_obs)
        rgb = out["left_image"]
        # Extracted left = 180°-rotate(channel-swap(bgra)) — both axes reversed.
        np.testing.assert_array_equal(rgb[..., 0], bgra[::-1, ::-1, 2])
        np.testing.assert_array_equal(rgb[..., 2], bgra[::-1, ::-1, 0])

    def test_left_external_image_is_rotated_180(self, warmed_obs, args):
        """First row of extracted left comes from last row of raw, read right-to-left."""
        imgs = warmed_obs["image"]
        key = next(k for k in imgs if args.left_camera_id in k and "left" in k)
        bgra = imgs[key]
        out = _extract_observation(args, warmed_obs)
        rgb = out["left_image"]
        # rgb[0, 0] came from bgra[-1, -1] (180° rotation).
        np.testing.assert_array_equal(rgb[0, :, 0], bgra[-1, ::-1, 2])
        np.testing.assert_array_equal(rgb[-1, :, 0], bgra[0, ::-1, 2])
        # And the top-left pixel of rgb == bottom-right of raw (after BGR→RGB).
        np.testing.assert_array_equal(rgb[0, 0, :], bgra[-1, -1, 2::-1])

    def test_wrist_image_is_not_flipped(self, warmed_obs, args):
        imgs = warmed_obs["image"]
        key = next(k for k in imgs if args.wrist_camera_id in k and "left" in k)
        bgra = imgs[key]
        out = _extract_observation(args, warmed_obs)
        rgb = out["wrist_image"]
        np.testing.assert_array_equal(rgb[..., 0], bgra[..., 2])
        np.testing.assert_array_equal(rgb[..., 2], bgra[..., 0])

    def test_full_pipeline_produces_valid_policy_input(self, warmed_obs, args, out_dir):
        out = _extract_observation(args, warmed_obs)
        ext_224 = image_tools.resize_with_pad(out["left_image"], 224, 224)
        wrist_224 = image_tools.resize_with_pad(out["wrist_image"], 224, 224)
        assert ext_224.shape == (224, 224, 3)
        assert wrist_224.shape == (224, 224, 3)
        assert ext_224.dtype == np.uint8
        Image.fromarray(ext_224).save(os.path.join(out_dir, "04_left_224_model_input.jpg"))
        Image.fromarray(wrist_224).save(os.path.join(out_dir, "05_wrist_224_model_input.jpg"))

    def test_trajectory_overlay_draws_on_real_frame(self, warmed_obs, args, out_dir):
        out = _extract_observation(args, warmed_obs)
        ext = out["left_image"]
        h, w = ext.shape[:2]
        pts = [[int(w * f), int(h * f)] for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
        annotated = _draw_trajectory_on_image(ext, pts, current_index=2, config=TraceOverlayConfig())
        assert annotated.shape == ext.shape
        Image.fromarray(annotated).save(os.path.join(out_dir, "06_left_with_fake_trajectory.jpg"))


# ---------------------------------------------------------------------------
# End-to-end trajectory generation with real Gemini + GPT
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HARDWARE_ENABLED, reason="not on robot laptop")
@pytest.mark.skipif(_MAIN_EXC is not None, reason=f"main.py import failed: {_MAIN_EXC}")
@pytest.mark.skipif(not _cams_ok, reason=f"ZED cameras not available: {_cams_reason}")
@pytest.mark.skipif(not _HAS_API_KEYS, reason="GEMINI_API_KEY and/or OPENAI_API_KEY not set")
class TestHardwareTrajectoryE2E:
    """Real end-to-end: live ZED frame + real Gemini + real GPT → real trajectory.

    Full pipeline:
      1. Capture frame from left ZED (via _extract_observation, so the flip is applied).
      2. GPT-4o-mini decomposes the instruction into manipulation steps.
      3. Plan-once merge (if >1 step).
      4. Gemini Robotics-ER locates the objects named in the step.
      5. GPT-4o-mini generates the trajectory waypoints.
      6. Scaled back to original image coordinates.

    The default instruction is "Pick up the marker and put it in the basket".
    Override via TEST_INSTRUCTION env var:

        TEST_INSTRUCTION="Pick up the red block" python -m pytest \
            tests/test_image_pipeline.py::TestHardwareTrajectoryE2E -v

    Outputs land in /tmp/test_image_pipeline/:
      07_real_trajectory_full_pipeline.jpg  — full-res overlay on the live frame
      08_real_trajectory_224_model_input.jpg — exactly what the policy would receive
    """

    DEFAULT_INSTRUCTION = "Pick up the marker and put it in the basket"

    @pytest.fixture
    def instruction(self):
        return os.environ.get("TEST_INSTRUCTION", self.DEFAULT_INSTRUCTION)

    def test_full_pipeline_generates_trajectory(
        self, instruction, args, warmed_obs, out_dir,
    ):
        # --- 1. Extract frame (with vertical flip applied) ---
        obs = _extract_observation(args, warmed_obs)
        ext_image = obs["left_image"]
        h, w = ext_image.shape[:2]

        # --- 2. Decompose instruction (real GPT call) ---
        pil = resize_for_api(Image.fromarray(ext_image).convert("RGB"))
        encoded = encode_pil_image(pil)
        decomp = query_target_objects(
            instruction, gpt_model=args.gpt_model, img_encoded=encoded,
        )
        steps = decomp.get("steps", [])
        assert steps, f"GPT decomposition returned no steps: {decomp}"
        for s in steps:
            for key in ("step", "manipulating_object", "target_related_object", "target_location"):
                assert key in s, f"decomposed step missing key {key!r}: {s}"

        # --- 3. Build planning state (with plan-once merge when multi-step) ---
        planning = PlanningState(full_task=instruction)
        planning.steps = steps
        if len(steps) > 1:
            first, last = steps[0], steps[-1]
            planning.steps = [{
                "step": instruction,
                "manipulating_object": first["manipulating_object"],
                "target_related_object": (
                    last.get("target_related_object") or first.get("target_related_object", "")
                ),
                "target_location": (
                    last.get("target_location") or first.get("target_location", "")
                ),
            }]

        # --- 4+5. Predict trajectory (real Gemini + GPT calls) ---
        traj = _predict_trajectory(planning, ext_image, args)
        assert traj is not None, "_predict_trajectory returned None"
        assert isinstance(traj, dict)
        for key in ("trajectory", "start_point", "end_point"):
            assert key in traj, f"trajectory dict missing key {key!r}: {list(traj.keys())}"

        pts = traj["trajectory"]
        assert len(pts) >= 3, f"too few waypoints: {len(pts)}"
        assert len(pts) <= 30, f"unexpectedly many waypoints: {len(pts)}"

        # --- Bounds check on trajectory coords (small tolerance for rounding) ---
        for i, p in enumerate(pts):
            x, y = p[0], p[1]
            assert -5 <= x <= w + 5, f"point {i} x={x} out of [0, {w}]"
            assert -5 <= y <= h + 5, f"point {i} y={y} out of [0, {h}]"

        # --- Meaningful displacement (not all clustered at one spot) ---
        sp, ep = traj["start_point"], traj["end_point"]
        disp = ((ep[0] - sp[0]) ** 2 + (ep[1] - sp[1]) ** 2) ** 0.5
        assert disp > 0.03 * max(w, h), (
            f"trajectory displacement too small: {disp:.1f}px (image {w}×{h})"
        )

        # --- 6. Save overlay outputs for visual inspection ---
        annotated = _draw_trajectory_on_image(
            ext_image, pts, current_index=1, config=TraceOverlayConfig(),
        )
        full_path = os.path.join(out_dir, "07_real_trajectory_full_pipeline.jpg")
        Image.fromarray(annotated).save(full_path)

        model_input = image_tools.resize_with_pad(annotated, 224, 224)
        model_path = os.path.join(out_dir, "08_real_trajectory_224_model_input.jpg")
        Image.fromarray(model_input).save(model_path)

        # --- Print diagnostics (visible with pytest -s or -v) ---
        print()
        print(f"  Instruction: {instruction!r}")
        print(f"  Decomposed into {len(steps)} step(s):")
        for i, s in enumerate(steps):
            print(f"    {i + 1}. {s['step']}")
            print(f"       manipulating={s['manipulating_object']!r}  "
                  f"target_related={s.get('target_related_object', '')!r}  "
                  f"target_location={s.get('target_location', '')!r}")
        if len(steps) > 1:
            merged = planning.steps[0]
            print(f"  [plan-once] merged → manipulate={merged['manipulating_object']!r}, "
                  f"target_related={merged['target_related_object']!r}, "
                  f"target_location={merged['target_location']!r}")
        print(f"  Trajectory: {len(pts)} waypoints")
        print(f"  Start:       ({sp[0]:.1f}, {sp[1]:.1f})")
        print(f"  End:         ({ep[0]:.1f}, {ep[1]:.1f})")
        print(f"  Displacement: {disp:.1f}px ({100 * disp / max(w, h):.1f}% of image)")
        reasoning = traj.get("reasoning", "(none)")
        print(f"  GPT reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
        if getattr(planning, "_last_object_locations", None):
            print(f"  Gemini detections:")
            for name, coord in planning._last_object_locations.items():
                print(f"    {name!r}: ({coord[0]:.1f}, {coord[1]:.1f})")
        print(f"  Saved:")
        print(f"    {full_path}")
        print(f"    {model_path}")
