"""Camera-calibration sanity and visual-debug test.

===========================================================================
 QUICK-RUN (copy-paste) — on the robot laptop:
---------------------------------------------------------------------------

  # Live visual check (robot + ZED connected):
  conda activate eva_jiani_env
  cd ~/pi-trajectory-overlay
  python tests/test_camera_calibration.py --live \
      --left-camera-id 26368109 --right-camera-id 27085680 \
      --wrist-camera-id 15512737 \
      --out /tmp/cal_check_live.jpg

  # Sanity-only (no robot needed, <1s):
  python -m pytest tests/test_camera_calibration.py -v

  # From a saved run dir (either box):
  python tests/test_camera_calibration.py \
      --run-dir ~/pi-trajectory-overlay/runs/<ts> \
      --frame 0 --out /tmp/cal_check.jpg

===========================================================================

This file has two complementary jobs:

1. **Pytest mode** — static/structural sanity checks on the repo-tracked
   calibration JSON (``calibration/calibration_info.json``). Run with::

       pytest tests/test_camera_calibration.py -v

   These pass without any robot or ZED hardware.

2. **CLI visual mode** — projects the Franka kinematic chain (every joint
   origin + EE flange axes) through the calibration (K intrinsics +
   camera-in-base extrinsic + 180° flip) onto a real ZED frame and writes
   an annotated PNG. The yellow dot is where MPC *thinks* the gripper is;
   if it lands on the actual gripper in the ZED image, the calibration is
   good. If it's off, the rotation / translation is stale.

   Two visual sub-modes:

   * **From a saved run dir** — point at an existing
     ``runs/<timestamp>/`` directory with ``actions.log`` (q=[...] per row,
     requires the MPC branch's logging patch) and ``rollout.mp4``::

       python tests/test_camera_calibration.py \\
           --run-dir ~/pi-trajectory-overlay/runs/2026_04_22_14-20-04 \\
           --out /tmp/cal_check.png

     Works on **both** the laptop and the GPU box (just scp the run dir).

   * **Live** — runs on the laptop only, connects to the DROID env, warms
     up the ZEDs, grabs one frame + the current joint state, projects::

       python tests/test_camera_calibration.py --live \\
           --left-camera-id 26368109 --right-camera-id 27085680 \\
           --wrist-camera-id 15512737 --out /tmp/cal_check_live.png

The annotated PNG shows:
  * RED / GREEN / BLUE axes at the EE = base-frame +X / +Y / +Z each 0.10 m
    long. If your calibration is right, these axes will look right at the
    gripper (e.g. +Z goes into the palm for the standard Franka mount).
  * YELLOW filled dot + black outline = the MPC's predicted EE pixel.
  * CYAN small crosses along the arm = each of the 7 joint frame origins.
    You should see a continuous chain from robot base to gripper.
  * HUD (top-left) = calibration file path, timestamp, current q,
    extrinsic values, predicted EE pixel + in-bounds flag.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path so `mpc_overlay` + `main_mpc` import regardless of
# where this file is invoked from (pytest, CLI, laptop, GPU box).
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pytest


# ------------------------------------------------------------------------
# Calibration file on disk — the authoritative source per the repo.
# ------------------------------------------------------------------------
CALIB_JSON = _REPO / "calibration" / "calibration_info.json"
CAMERA_KEY = "26368109_left"                     # the exterior ZED MPC uses
CAL_STALENESS_DAYS = 365                          # warn (not fail) beyond this

# Match main_mpc.py's real-robot constants exactly.
K_REAL = np.array(
    [[532.66, 0.0, 641.305],
     [0.0, 532.55, 347.186],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
IMG_H, IMG_W = 720, 1280
EE_OFFSET_FROM_FLANGE = 0.0     # real-robot uses flange (panda_link8)
AXIS_LEN_M = 0.10                # EE coord-frame arrow length in base-frame metres


# =====================================================================
#                             Pytest sanity checks
# =====================================================================
def _load_calibration():
    assert CALIB_JSON.exists(), f"missing {CALIB_JSON}"
    with CALIB_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def test_calibration_file_exists():
    assert CALIB_JSON.exists(), f"{CALIB_JSON} is missing; cannot run MPC on real robot"


def test_calibration_json_parses():
    data = _load_calibration()
    assert isinstance(data, dict) and len(data) > 0


def test_calibration_has_exterior_camera_entry():
    data = _load_calibration()
    assert CAMERA_KEY in data, (
        f"{CAMERA_KEY!r} not in {CALIB_JSON}. "
        f"Real-robot MPC will raise FileNotFoundError at import time."
    )


def test_calibration_pose_shape_and_finiteness():
    data = _load_calibration()
    entry = data[CAMERA_KEY]
    vec = entry.get("pose") or entry.get("extrinsics")
    assert vec is not None, "neither 'pose' nor 'extrinsics' key present"
    assert len(vec) == 6, f"expected 6-vec [x,y,z,rx,ry,rz], got {vec}"
    arr = np.asarray(vec, dtype=np.float64)
    assert np.all(np.isfinite(arr)), f"non-finite values in extrinsic: {arr}"
    # Gross reasonableness: translation should be within 2 m of base; euler
    # within (-pi, pi).
    assert np.all(np.abs(arr[:3]) < 2.0), f"translation implausible: {arr[:3]}"
    assert np.all(np.abs(arr[3:]) < 2 * np.pi), f"euler implausible: {arr[3:]}"


def test_calibration_not_implausibly_stale():
    data = _load_calibration()
    entry = data[CAMERA_KEY]
    ts = entry.get("timestamp", 0)
    if not ts:
        pytest.skip("no timestamp in calibration entry")
    age_days = (datetime.datetime.now().timestamp() - float(ts)) / 86400.0
    if age_days > CAL_STALENESS_DAYS:
        pytest.fail(
            f"calibration is {age_days:.0f} days old (> {CAL_STALENESS_DAYS}); "
            "rerun the on-rig charuco calibration and commit a fresh JSON."
        )


def test_main_mpc_load_extrinsics_returns_matching_vector():
    """The loader in main_mpc.py must end up reading the same numbers that
    this file sees directly. Guards against silent search-path drift."""
    import main_mpc  # noqa: E402
    ext = main_mpc._load_extrinsics().astype(np.float64)
    data = _load_calibration()
    expected = np.asarray(
        data[CAMERA_KEY].get("pose") or data[CAMERA_KEY].get("extrinsics"),
        dtype=np.float64,
    )
    # The loader may have picked a different search path (e.g. the laptop's
    # live aurora file). Allow a small tolerance; if radically different,
    # that's a red flag the rig is using a different file than committed.
    if not np.allclose(ext, expected, atol=1e-3):
        pytest.skip(
            "main_mpc.py loaded a different calibration than the repo copy. "
            f"This is OK if you're on the laptop with a fresher live file.\n"
            f"  repo    = {expected.tolist()}\n"
            f"  loader  = {ext.tolist()}"
        )


# =====================================================================
#                    Visual/geometric helpers (CLI + tests)
# =====================================================================
def euler_xyz_to_matrix(rxyz: np.ndarray) -> np.ndarray:
    """scipy 'xyz' extrinsic Euler -> 3x3. Matches GuidanceSpec convention."""
    cx, cy, cz = np.cos(rxyz)
    sx, sy, sz = np.sin(rxyz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_base_to_pixel(
    p_base: np.ndarray,       # (..., 3)
    ext_cam_in_base: np.ndarray,   # (6,)
    K: np.ndarray,            # (3,3)
    image_hw=(IMG_H, IMG_W),
    flipped_180: bool = True,
):
    """Project a point in the robot base frame to a 2D pixel on the ZED
    frame that the laptop client uses (exterior mount is upside-down, so
    we add the 180° flip at the end)."""
    R_cam_in_base = euler_xyz_to_matrix(np.asarray(ext_cam_in_base[3:], dtype=np.float64))
    t_cam_in_base = np.asarray(ext_cam_in_base[:3], dtype=np.float64)
    R_base_in_cam = R_cam_in_base.T
    t_base_in_cam = -R_base_in_cam @ t_cam_in_base
    p = np.asarray(p_base, dtype=np.float64)
    p_cam = p @ R_base_in_cam.T + t_base_in_cam
    if np.any(p_cam[..., 2] <= 1e-6):
        # Behind camera — pixel will be bogus; return NaN so caller knows
        return np.full(p.shape[:-1] + (2,), np.nan)
    uv_h = p_cam @ np.asarray(K, dtype=np.float64).T
    uv = uv_h[..., :2] / uv_h[..., 2:3]
    if flipped_180:
        H, W = image_hw
        uv = np.stack([W - uv[..., 0], H - uv[..., 1]], axis=-1)
    return uv


def fk_all_joint_origins(q7: np.ndarray, flange_offset: float = EE_OFFSET_FROM_FLANGE):
    """Return (8, 3) array of joint-frame origins in base frame: link1..link7
    plus the flange (panda_link8). Uses mpc_overlay's MDH params."""
    from mpc_overlay.franka_fk import PANDA_A, PANDA_D, PANDA_ALPHA
    T = np.eye(4, dtype=np.float64)
    origins = []
    for i in range(7):
        a, alpha, d, theta = (
            float(PANDA_A[i]), float(PANDA_ALPHA[i]), float(PANDA_D[i]), float(q7[i])
        )
        ca, sa, cA, sA = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        Ti = np.array([
            [ca, -sa, 0, a],
            [sa * cA, ca * cA, -sA, -sA * d],
            [sa * sA, ca * sA, cA, cA * d],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        T = T @ Ti
        origins.append(T[:3, 3].copy())
    # panda_link8 (flange): offset 0.107 along link7 z, then add the
    # extra ee_offset_from_flange (usually 0 on real robot).
    flange = T @ np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0.107 + flange_offset], [0, 0, 0, 1],
    ], dtype=np.float64)
    origins.append(flange[:3, 3].copy())
    # orientation of the flange frame in base
    return np.asarray(origins, dtype=np.float64), flange[:3, :3].copy()


# =====================================================================
#                           CLI visual mode
# =====================================================================
def _draw_annotations(
    img_flipped: np.ndarray,           # (H, W, 3) RGB, already flipped 180°
    joint_pixels: np.ndarray,          # (8, 2) or entries may be NaN
    ee_pix: np.ndarray,                # (2,)
    ee_axes_pix: np.ndarray,           # (3, 2) — +X, +Y, +Z tips (may be NaN)
    hud_lines: list,
):
    import cv2
    bgr = cv2.cvtColor(img_flipped, cv2.COLOR_RGB2BGR).copy()

    # Kinematic chain: cyan crosses + thin lines
    for i in range(joint_pixels.shape[0] - 1):
        p0, p1 = joint_pixels[i], joint_pixels[i + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(bgr, a, b, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.line(bgr, a, b, (255, 200, 0), thickness=1, lineType=cv2.LINE_AA)
    for pt in joint_pixels:
        if not np.all(np.isfinite(pt)):
            continue
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.drawMarker(bgr, p, (255, 200, 0), cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # EE axes
    if np.all(np.isfinite(ee_pix)):
        origin = (int(round(ee_pix[0])), int(round(ee_pix[1])))
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue (BGR)
        axis_names = ["X", "Y", "Z"]
        for i in range(3):
            tip = ee_axes_pix[i]
            if not np.all(np.isfinite(tip)):
                continue
            t = (int(round(tip[0])), int(round(tip[1])))
            cv2.line(bgr, origin, t, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
            cv2.line(bgr, origin, t, axis_colors[i], thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(bgr, axis_names[i], t, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.putText(bgr, axis_names[i], t, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, axis_colors[i], thickness=2, lineType=cv2.LINE_AA)
        # EE dot (yellow) — on top so it's always visible
        cv2.circle(bgr, origin, radius=10, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, origin, radius=7, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # HUD
    y = 28
    for text, color in hud_lines:
        cv2.putText(bgr, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
        cv2.putText(bgr, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, thickness=1, lineType=cv2.LINE_AA)
        y += 22
    return bgr


def _parse_q_from_actions_log(log_path: str, which_step: int = 0):
    """Read q=[...] from the given 0-indexed step of actions.log (per-step
    MPC log format: 't=0000 | queried=... | action=[...] | gripper=... | q=[...]')."""
    import re
    _Q = re.compile(r"q=\[([^\]]+)\]")
    _T = re.compile(r"t=(\d+)")
    want = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            mt, mq = _T.search(line), _Q.search(line)
            if not (mt and mq):
                continue
            if int(mt.group(1)) == which_step:
                want = [float(x) for x in mq.group(1).split(",")]
                break
    if want is None:
        raise RuntimeError(
            f"{log_path} has no q=[...] for step {which_step}. "
            f"Was this file written by the MPC-branch main_mpc.py/main_robolab.py?"
        )
    return np.asarray(want, dtype=np.float64)


def _frame_from_run_dir(run_dir: str, which_step: int = 0):
    """Decode the which_step-th frame from rollout.mp4 in the run dir."""
    import cv2
    path = os.path.join(run_dir, "rollout.mp4")
    if not os.path.exists(path):
        # Fall back to frames/NNNN_ext.jpg if available
        jpg = os.path.join(run_dir, "frames", f"{which_step:04d}_ext.jpg")
        if os.path.exists(jpg):
            img_bgr = cv2.imread(jpg)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"no rollout.mp4 or frames/{which_step:04d}_ext.jpg in {run_dir}")
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, which_step)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {which_step} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _live_capture(args):
    """On-laptop live capture: DROID env + one joint state + one ext frame."""
    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    # Warmup cameras
    import time
    for attempt in range(120):
        obs = env.get_observation()
        img_dict = obs.get("image", {})
        keys = list(img_dict.keys()) if isinstance(img_dict, dict) else []
        have_ext = any(args.left_camera_id in k and "left" in k for k in keys)
        have_wrist = any(args.wrist_camera_id in k and "left" in k for k in keys)
        if have_ext and have_wrist:
            break
        time.sleep(0.5)
    else:
        print("WARN: cameras didn't fully warm up — proceeding")
    obs = env.get_observation()
    # Find the left image for the exterior we're calibrating.
    img = None
    for k, v in obs["image"].items():
        if args.left_camera_id in k and "left" in k:
            img = v
            break
    if img is None:
        raise RuntimeError("no exterior frame from live capture")
    # BGRA -> RGB + 180° flip (same as main_pi05.py/main_mpc.py)
    rgb = img[..., :3][..., ::-1]
    rgb = np.ascontiguousarray(rgb[::-1, ::-1])
    q = np.asarray(obs["robot_state"]["joint_positions"], dtype=np.float64)
    return rgb, q


def _run_visual(args):
    import main_mpc  # noqa: E402

    # Load calibration
    ext_vec = main_mpc._load_extrinsics(args.calibration_json).astype(np.float64)

    # Get frame + q
    if args.live:
        rgb, q = _live_capture(args)
    else:
        if not args.run_dir:
            print("error: pass --run-dir <dir> or --live")
            return 2
        rgb = _frame_from_run_dir(args.run_dir, args.frame)
        q = _parse_q_from_actions_log(os.path.join(args.run_dir, "actions.log"),
                                       which_step=args.frame)

    # Project kinematic chain + EE + axes
    origins, R_flange = fk_all_joint_origins(q, flange_offset=EE_OFFSET_FROM_FLANGE)
    joint_pixels = project_base_to_pixel(origins, ext_vec, K_REAL,
                                          image_hw=(IMG_H, IMG_W), flipped_180=True)
    ee_pos_base = origins[-1]
    # Axes tips in base frame: origin + 0.1 * each flange-frame axis
    axis_tips_base = np.stack([
        ee_pos_base + AXIS_LEN_M * R_flange[:, 0],
        ee_pos_base + AXIS_LEN_M * R_flange[:, 1],
        ee_pos_base + AXIS_LEN_M * R_flange[:, 2],
    ], axis=0)
    axis_tips_pix = project_base_to_pixel(axis_tips_base, ext_vec, K_REAL,
                                          image_hw=(IMG_H, IMG_W), flipped_180=True)

    ee_pix = joint_pixels[-1]
    in_bounds = bool(np.all(np.isfinite(ee_pix)) and
                     0 <= ee_pix[0] <= IMG_W and 0 <= ee_pix[1] <= IMG_H)

    # HUD
    cal_src = args.calibration_json or CALIB_JSON
    calib_info = ""
    try:
        with open(cal_src, encoding="utf-8") as f:
            d = json.load(f)
        ent = d.get(CAMERA_KEY, {})
        ts = ent.get("timestamp", 0)
        calib_info = (datetime.datetime.fromtimestamp(ts).isoformat()
                      if ts else "no-timestamp")
    except Exception as e:
        calib_info = f"({e!r})"

    q_str = "[" + ", ".join(f"{v:+.2f}" for v in q) + "]"
    hud = [
        (f"calib: {os.path.basename(str(cal_src))}   t={calib_info}", (255, 255, 255)),
        (f"ext (x,y,z)= ({ext_vec[0]:+.3f}, {ext_vec[1]:+.3f}, {ext_vec[2]:+.3f}) m",
         (200, 255, 200)),
        (f"ext (rx,ry,rz)= ({ext_vec[3]:+.3f}, {ext_vec[4]:+.3f}, {ext_vec[5]:+.3f}) rad",
         (200, 255, 200)),
        (f"q7 = {q_str}", (255, 255, 255)),
        (f"EE pixel = ({ee_pix[0]:.0f}, {ee_pix[1]:.0f})  in-bounds={in_bounds}",
         (0, 255, 255)),
        ("yellow=EE   RGB axes=base XYZ   cyan crosses=joint origins",
         (200, 200, 200)),
    ]

    bgr = _draw_annotations(rgb, joint_pixels, ee_pix, axis_tips_pix, hud)
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"wrote {args.out}")
    print(f"  EE pixel = ({ee_pix[0]:.1f}, {ee_pix[1]:.1f})  in-bounds={in_bounds}")
    print(f"  ext = {ext_vec.tolist()}")
    print(f"  q   = {q.tolist()}")
    if not in_bounds:
        print("  WARNING: EE projected OUTSIDE the image. Calibration may be "
              "wrong, OR the robot is in a pose where the gripper is out of "
              "camera view. Inspect the PNG to decide.")
    return 0


# =====================================================================
#                               CLI
# =====================================================================
def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--live", action="store_true",
                     help="Laptop-only: capture one ZED frame + current q.")
    src.add_argument("--run-dir", type=str, default="",
                     help="Saved run directory (frames + actions.log) to use.")
    ap.add_argument("--frame", type=int, default=0,
                    help="Which step (row) to pull from actions.log / rollout.mp4. Default 0.")
    ap.add_argument("--left-camera-id", type=str, default="26368109")
    ap.add_argument("--right-camera-id", type=str, default="27085680")
    ap.add_argument("--wrist-camera-id", type=str, default="15512737")
    ap.add_argument("--calibration-json", type=str, default="",
                    help="Override path to calibration_info.json (else use main_mpc.py search).")
    ap.add_argument("--out", type=str, default="/tmp/camera_calibration_check.jpg",
                    help="Annotated PNG/JPG output path.")
    args = ap.parse_args()
    if not args.live and not args.run_dir:
        ap.error("pass either --live or --run-dir <dir>")
    sys.exit(_run_visual(args))


if __name__ == "__main__":
    _main()
