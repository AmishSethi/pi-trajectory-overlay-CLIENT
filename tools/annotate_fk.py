"""Render annotated_rollout.mp4 for an MPC run using the LOGGED per-step joint
state so the yellow dot sits on the ACTUAL predicted EE pixel — not a
time-based stand-in.

Inputs (all produced by simulator/main_robolab.py with guidance-mode=mpc):
  <run_dir>/rollout.mp4                       raw external-cam frames
  <run_dir>/actions.log                       per-step rows with q=[...]
  <run_dir>/mpc_waypoints[_NNN].json          one file per active plan
  <run_dir>/guidance_spec_snapshot.json       intrinsics / extrinsic / calibration

Output:
  <out_path>   annotated mp4 with:
      - red->pink gradient arrow (matches the training-time TraceOverlayConfig)
      - yellow dot = FK-projected EE at the logged q_t (what MPC actually saw)
      - blue cross = the arrow point at the EE's current arc-length projection
        (s0), so you can see how far along the arrow the MPC thinks we are
      - HUD: step count, plan idx + t_plan, s0_frac %, flash on re-plan
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch

# Make mpc_overlay importable regardless of install location: add the
# pi-trajectory-overlay repo root (parent of this file's "tools/" dir)
# to sys.path. Works on the GPU box (asethi04) and the robot laptop (franka).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mpc_overlay import GuidanceSpec  # noqa: E402
from mpc_overlay.trajectory_cost import (  # noqa: E402
    _cumulative_arc_length,
    _project_ee_to_arc,
    ee_pixel_at_q0,
)


# Regex to pull `q=[...]` out of an actions.log line.
_Q_RX = re.compile(r"q=\[([^\]]+)\]")
_T_RX = re.compile(r"t=(\d+)")


@dataclass
class WpWindow:
    t_step: int
    waypoints: np.ndarray


def parse_actions_log(path: str) -> List[np.ndarray]:
    """Return a list of q-vectors indexed by t_step. Missing rows → None entries."""
    per_step = {}
    with open(path) as f:
        for line in f:
            m_t = _T_RX.search(line)
            m_q = _Q_RX.search(line)
            if not m_t or not m_q:
                continue
            t = int(m_t.group(1))
            q = np.asarray([float(x) for x in m_q.group(1).split(",")], dtype=np.float32)
            per_step[t] = q
    if not per_step:
        return []
    T = max(per_step.keys()) + 1
    out: List[np.ndarray] = [per_step.get(t) for t in range(T)]
    return out


def load_spec_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_waypoint_windows(run_dir: str) -> List[WpWindow]:
    json_paths = sorted(glob.glob(os.path.join(run_dir, "mpc_waypoints*.json")))
    windows: List[WpWindow] = []
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        t = int(d.get("t_step", 0))
        wp = np.asarray(d["waypoints_px"], dtype=np.float32)
        windows.append(WpWindow(t_step=t, waypoints=wp))
    return windows


def active_window(windows: List[WpWindow], t: int):
    active = None
    for w in windows:
        if w.t_step <= t:
            active = w
        else:
            break
    return active


def make_spec(snapshot: dict, q_t: np.ndarray, waypoints: np.ndarray) -> GuidanceSpec:
    """Re-assemble a GuidanceSpec the annotator can feed to the FK projector."""
    K = torch.tensor(snapshot["K_intrinsics"], dtype=torch.float32)
    ext = torch.tensor(snapshot["extrinsic_cam_in_base"], dtype=torch.float32)
    jvs = torch.tensor(snapshot["joint_vel_scale"], dtype=torch.float32)
    wp = torch.tensor(waypoints, dtype=torch.float32)
    q0 = torch.tensor(q_t[:7], dtype=torch.float32).unsqueeze(0)
    return GuidanceSpec(
        waypoints_px=wp,
        image_hw=tuple(snapshot["image_hw"]),
        K_intrinsics=K,
        extrinsic_cam_in_base=ext,
        image_flipped_180=bool(snapshot["image_flipped_180"]),
        control_dt=float(snapshot["control_dt"]),
        joint_vel_scale=jvs,
        q0=q0,
        ee_offset_from_flange=float(snapshot["ee_offset_from_flange"]),
    )


# --- drawing helpers -----------------------------------------------------------
def draw_arrow(img: np.ndarray, wp: np.ndarray) -> np.ndarray:
    if wp is None or wp.shape[0] < 2:
        return img
    out = img.copy()
    n = wp.shape[0]
    pts = [(int(round(u)), int(round(v))) for u, v in wp]
    # outline
    for i in range(n - 1):
        cv2.line(out, pts[i], pts[i + 1], (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    red = np.array([255, 0, 0], dtype=np.float32)    # RGB
    pink = np.array([255, 105, 180], dtype=np.float32)
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        rgb = ((1 - t) * red + t * pink).astype(np.int32).tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.line(out, pts[i], pts[i + 1], bgr, thickness=3, lineType=cv2.LINE_AA)
    return out


def draw_ee_dot(img: np.ndarray, ee_px: tuple) -> None:
    """Yellow dot with black outline at the ACTUAL EE pixel."""
    u, v = int(round(ee_px[0])), int(round(ee_px[1]))
    cv2.circle(img, (u, v), radius=9, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, (u, v), radius=6, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def draw_arrow_proj(img: np.ndarray, proj_px: tuple) -> None:
    """Blue cross at the current arc-length projection of the EE onto the arrow."""
    u, v = int(round(proj_px[0])), int(round(proj_px[1]))
    L = 14
    cv2.line(img, (u - L, v), (u + L, v), (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    cv2.line(img, (u, v - L), (u, v + L), (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    cv2.line(img, (u - L, v), (u + L, v), (255, 128, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.line(img, (u, v - L), (u, v + L), (255, 128, 0), thickness=3, lineType=cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: List[tuple]) -> None:
    """lines: [(text, color_bgr), ...] drawn top-down from (15, 30)."""
    y = 30
    for text, color in lines:
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 28


def annotate(run_dir: str, out_path: str, task_label: str) -> None:
    video = os.path.join(run_dir, "rollout.mp4")
    log = os.path.join(run_dir, "actions.log")
    snap_path = os.path.join(run_dir, "guidance_spec_snapshot.json")
    q_list = parse_actions_log(log)
    snapshot = load_spec_snapshot(snap_path) if os.path.exists(snap_path) else None
    windows = load_waypoint_windows(run_dir)
    # Three independent capabilities — we can do partial annotation when
    # some are missing. This lets the annotator serve both as an
    # MPC-rollout viz AND as a whole-video calibration checker on baseline
    # (no-MPC) runs that still have q_t logged and a calibration snapshot.
    has_fk = bool(q_list) and snapshot is not None
    has_arrow = bool(windows)
    if has_fk and has_arrow:
        mode_tag = "[MPC]"
    elif has_fk:
        mode_tag = "[FK-only]"    # draw EE dot only, no arrow
    else:
        mode_tag = "[baseline]"   # no EE, no arrow — HUD + grip flash only
    if not has_fk:
        print(f"  [annotate] no FK data (q-log or spec snapshot missing); "
              f"will write HUD-only overlay")
    elif not has_arrow:
        print(f"  [annotate] no MPC waypoints in run dir; drawing EE dot each "
              f"frame for calibration verification, no arrow overlay")

    cap = cv2.VideoCapture(video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    last_plan_t = None
    flash = 0
    # Pre-parse grip events for HUD annotation (helpful in baseline too).
    grip_events = []  # list of (t, "CLOSE"/"OPEN")
    log_grippers = []
    if os.path.exists(log):
        prev = None
        with open(log) as f:
            for line in f:
                m_t = _T_RX.search(line)
                m_g = re.search(r"gripper=([-\d\.]+)", line)
                if not m_t or not m_g:
                    continue
                tt = int(m_t.group(1))
                g = float(m_g.group(1))
                log_grippers.append((tt, g))
                if prev is not None:
                    if prev < 0.5 and g >= 0.5:
                        grip_events.append((tt, "CLOSE"))
                    elif prev >= 0.5 and g < 0.5:
                        grip_events.append((tt, "OPEN"))
                prev = g
    grip_by_t = {t: g for t, g in log_grippers}
    grip_flash = {}  # step -> remaining frames to flash the event badge
    for t, ev in grip_events:
        grip_flash[t] = ev

    # Dummy 2-point waypoint used to build a GuidanceSpec when the run has
    # FK data but no MPC waypoints. The spec's waypoints_px field doesn't
    # affect ee_pixel_at_q0 — only q0/K/ext/jvs/flange-offset matter for it.
    _DUMMY_WP = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    for t in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        w = active_window(windows, t) if has_arrow else None
        q_t = q_list[t] if (has_fk and t < len(q_list)) else None
        if w is not None and w.t_step != last_plan_t:
            last_plan_t = w.t_step
            flash = 8
        out = frame
        hud = [(f"{mode_tag} {task_label}  step {t}/{n_frames - 1}", (255, 255, 255))]
        cur_g = grip_by_t.get(t, None)
        if cur_g is not None:
            if cur_g >= 0.5:
                hud.append(("gripper: CLOSED", (64, 200, 64)))
            else:
                hud.append(("gripper: open", (200, 200, 200)))
        # grip-event flash: show "CLOSE" / "OPEN" for ~8 frames
        ev_recent = None
        for dt in range(8):
            if (t - dt) in grip_flash:
                ev_recent = (grip_flash[t - dt], dt)
                break
        if ev_recent is not None:
            ev, age = ev_recent
            col = (0, 255, 255) if ev == "CLOSE" else (255, 200, 0)
            cv2.putText(out, f"** GRIP {ev} **", (int(W*0.35), 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(out, f"** GRIP {ev} **", (int(W*0.35), 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, col, 2, cv2.LINE_AA)

        # Arrow overlay (requires waypoints for this frame).
        if w is not None:
            out = draw_arrow(out, w.waypoints)
            hud.append((f"arrow plan #{windows.index(w) + 1}  (t_plan={w.t_step})",
                        (0, 255, 255)))

        # EE dot (requires FK data — q_t + snapshot). Drawn whether or not we
        # also have waypoints, so the annotator doubles as a whole-video
        # calibration checker on baseline runs.
        if has_fk and q_t is not None and q_t.shape[0] >= 7:
            wp_for_spec = w.waypoints if w is not None else _DUMMY_WP
            spec = make_spec(snapshot, q_t, wp_for_spec)
            with torch.no_grad():
                ee_px = ee_pixel_at_q0(spec).cpu().numpy()
            # If we also have waypoints, draw projection cross + dashed
            # off-arrow line FIRST so the yellow EE dot ends up on top.
            if w is not None:
                wp_t = torch.tensor(w.waypoints, dtype=torch.float32)
                cum, total = _cumulative_arc_length(wp_t)
                s_frac = 0.0
                if float(total) > 0.0:
                    ee_tensor = torch.tensor(ee_px, dtype=torch.float32)
                    s0 = _project_ee_to_arc(ee_tensor, wp_t, cum, total)
                    s_frac = float(s0 / total)
                    seg = wp_t[1:] - wp_t[:-1]
                    seg_len = torch.linalg.vector_norm(seg, dim=-1)
                    cumlen = torch.cat([torch.zeros(1), torch.cumsum(seg_len, dim=0)])
                    idx = int(torch.searchsorted(cumlen, s0, right=False).clamp(
                        min=1, max=wp_t.shape[0] - 1).item())
                    s_before = float(cumlen[idx - 1])
                    span = float(cumlen[idx] - cumlen[idx - 1])
                    alpha = 0.0 if span < 1e-9 else float((s0 - s_before) / span)
                    alpha = max(0.0, min(1.0, alpha))
                    proj_px = wp_t[idx - 1] + alpha * (wp_t[idx] - wp_t[idx - 1])
                    draw_arrow_proj(out, (float(proj_px[0]), float(proj_px[1])))
                    p1 = (int(round(float(ee_px[0]))), int(round(float(ee_px[1]))))
                    p2 = (int(round(float(proj_px[0]))), int(round(float(proj_px[1]))))
                    cv2.line(out, p1, p2, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                    cv2.line(out, p1, p2, (255, 200, 0), thickness=1, lineType=cv2.LINE_AA)
                    hud.append((f"s0 = {s_frac * 100:.1f}% of arrow", (255, 128, 0)))
                    hud.append(("yellow=EE(actual)  blue+=projection", (255, 255, 255)))
            else:
                # Calibration-check mode: annotate what the dot MEANS since
                # there's no arrow context to infer it from.
                hud.append((f"yellow=EE(FK) @ ({ee_px[0]:.0f},{ee_px[1]:.0f})  "
                            f"— should sit on the real gripper", (0, 255, 255)))
            draw_ee_dot(out, (float(ee_px[0]), float(ee_px[1])))
        if flash > 0:
            cv2.rectangle(out, (0, 0), (W - 1, H - 1), (0, 255, 255), thickness=4)
            cv2.putText(out, "** RE-PLAN **", (W - 240, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, "** RE-PLAN **", (W - 240, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            flash -= 1
        draw_hud(out, hud)
        writer.write(out)
    cap.release()
    writer.release()
    print(f"wrote {out_path}  ({n_frames} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    # Sanitise the label so opencv's non-unicode font doesn't render glyphs as "?".
    safe = (args.label or os.path.basename(args.run_dir))
    safe = safe.replace("λ", "lam_").replace("σ", "sig_")
    annotate(args.run_dir, args.out, safe)
