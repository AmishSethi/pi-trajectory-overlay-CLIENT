"""
Trajectory visualization utilities for drawing smooth overlays on images.

Provides `add_trace_overlay` with `TraceOverlayConfig` for configurable
gradient lines, arrows, dots, and outlines.

IMPORTANT: The default TraceOverlayConfig matches the config used during
training of the pi05_droid_trajectory_overlay checkpoint. Do NOT change
the defaults without retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep


@dataclass
class TraceOverlayConfig:
    """All visual parameters for trace overlay rendering.

    The defaults here match the config used during training of the
    pi05_droid_trajectory_overlay checkpoint (from build_dataset.py in
    github.com/branyang02/temp_drawing/droid_trajectory):
      - Red-to-pink gradient line with black outline
      - Yellow dot at current position with black outline
      - No spline interpolation (use num_interpolated=0)
    """

    horizon: int = 0
    show_past: bool = False
    past_horizon: int = 0

    future_color: Tuple[int, int, int] = (255, 0, 0)
    future_color_end: Optional[Tuple[int, int, int]] = (255, 105, 180)
    future_outline_color: Tuple[int, int, int] = (0, 0, 0)
    future_thickness: int = 3
    future_outline_thickness: int = 5

    past_color: Tuple[int, int, int] = (180, 180, 180)
    past_color_end: Optional[Tuple[int, int, int]] = None
    past_outline_color: Tuple[int, int, int] = (0, 0, 0)
    past_thickness: int = 1
    past_outline_thickness: int = 0

    current_dot_radius: int = 5
    current_dot_color: Tuple[int, int, int] = (255, 255, 0)
    current_dot_outline_color: Tuple[int, int, int] = (0, 0, 0)
    current_dot_outline_thickness: int = 2

    use_alpha: bool = False
    alpha: float = 1.0

    dashed_future: bool = False
    dash_len: int = 10
    gap_len: int = 6

    arrow_mode: Literal["end_only", "multiple"] = "end_only"
    arrow_count: int = 0
    arrow_size: int = 12
    arrow_thickness: int = 1
    arrow_color: Tuple[int, int, int] = (0, 255, 255)
    arrow_outline_color: Tuple[int, int, int] = (0, 0, 0)
    arrow_outline_thickness: int = 0

    tick_marks: bool = False
    tick_every: int = 5
    tick_radius: int = 2
    tick_color: Tuple[int, int, int] = (255, 255, 255)
    tick_outline_color: Tuple[int, int, int] = (0, 0, 0)


def _lerp_color(
    c0: Tuple[int, int, int],
    c1: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return (
        round(c0[0] + (c1[0] - c0[0]) * t),
        round(c0[1] + (c1[1] - c0[1]) * t),
        round(c0[2] + (c1[2] - c0[2]) * t),
    )


def _draw_dashed_polyline_cv(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int,
    gap_len: int,
    color_end: Optional[Tuple[int, int, int]] = None,
) -> None:
    if len(pts) < 2:
        return

    total_len = 0.0
    if color_end is not None:
        for i in range(len(pts) - 1):
            total_len += math.hypot(
                float(pts[i + 1][0]) - float(pts[i][0]),
                float(pts[i + 1][1]) - float(pts[i][1]),
            )
        if total_len < 1e-6:
            return

    cycle = dash_len + gap_len
    accum = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-6:
            continue
        dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
        d = 0.0
        while d < seg_len:
            phase = (accum + d) % cycle
            if phase < dash_len:
                dash_remaining = dash_len - phase
                end_d = min(d + dash_remaining, seg_len)
                p0 = (round(x0 + dx * d), round(y0 + dy * d))
                p1 = (round(x0 + dx * end_d), round(y0 + dy * end_d))
                c = _lerp_color(color, color_end, (accum + d) / total_len) if color_end is not None else color
                cv2.line(img, p0, p1, c, thickness, cv2.LINE_AA)
                d = end_d
            else:
                d += cycle - phase
        accum += seg_len


def _draw_polyline_with_outline(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    outline_color: Tuple[int, int, int],
    outline_thickness: int,
    *,
    color_end: Optional[Tuple[int, int, int]] = None,
    is_dashed: bool = False,
    dash_len: int = 10,
    gap_len: int = 6,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if len(pts) < 2:
        return img
    overlay = img.copy() if alpha < 1.0 else img
    if is_dashed:
        if outline_thickness > 0:
            _draw_dashed_polyline_cv(overlay, pts, outline_color, outline_thickness, dash_len, gap_len)
        _draw_dashed_polyline_cv(overlay, pts, color, thickness, dash_len, gap_len, color_end=color_end)
    elif color_end is not None:
        np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if outline_thickness > 0:
            cv2.polylines(overlay, [np_pts], False, outline_color, outline_thickness, cv2.LINE_AA, shift=shift)
        n = len(pts) - 1
        for i in range(n):
            t = i / max(n - 1, 1)
            seg_color = _lerp_color(color, color_end, t)
            seg = np.array([pts[i], pts[i + 1]], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [seg], False, seg_color, thickness, cv2.LINE_AA, shift=shift)
    else:
        np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if outline_thickness > 0:
            cv2.polylines(overlay, [np_pts], False, outline_color, outline_thickness, cv2.LINE_AA, shift=shift)
        cv2.polylines(overlay, [np_pts], False, color, thickness, cv2.LINE_AA, shift=shift)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _draw_circle_with_outline(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    outline_thickness: int,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if radius <= 0:
        return img
    scale = 1 << shift
    overlay = img.copy() if alpha < 1.0 else img
    if outline_thickness > 0:
        cv2.circle(
            overlay, center, radius + outline_thickness * scale // 2,
            outline_color, thickness=-1, lineType=cv2.LINE_AA, shift=shift,
        )
    cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA, shift=shift)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _draw_arrows_with_outline(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    indices: list[int],
    color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    thickness: int,
    outline_thickness: int,
    tip_length_px: int,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if len(pts) < 2 or not indices:
        return img
    overlay = img.copy() if alpha < 1.0 else img
    for idx in indices:
        if idx < 0 or idx + 1 >= len(pts):
            continue
        p0, p1 = pts[idx], pts[idx + 1]
        seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_len < 1e-3:
            continue
        scale = 1 << shift
        tip_ratio = min(tip_length_px * scale / seg_len, 0.5)
        if outline_thickness > 0:
            cv2.arrowedLine(overlay, p0, p1, outline_color, outline_thickness, cv2.LINE_AA, shift=shift, tipLength=tip_ratio)
        cv2.arrowedLine(overlay, p0, p1, color, thickness, cv2.LINE_AA, shift=shift, tipLength=tip_ratio)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _arrow_indices(future_len: int, cfg: TraceOverlayConfig) -> list[int]:
    if future_len < 2 or cfg.arrow_count == 0:
        return []
    if cfg.arrow_mode == "end_only":
        return [future_len - 2]
    count = min(cfg.arrow_count, future_len - 1)
    if count <= 0:
        return []
    step = (future_len - 1) / (count + 1)
    return [round(step * (i + 1)) for i in range(count)]


def _annotate_frame(
    frame_rgb: np.ndarray,
    pts: list[Tuple[int, int]],
    t: int,
    cfg: TraceOverlayConfig,
    shift: int = 0,
) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return frame_rgb

    t = max(0, min(t, n - 1))
    alpha = cfg.alpha if cfg.use_alpha else 1.0

    future_end = n if cfg.horizon <= 0 else min(t + 1 + cfg.horizon, n)
    future_pts = pts[t:future_end]

    past_pts: list[Tuple[int, int]] = []
    if cfg.show_past and t > 0:
        past_start = 0 if cfg.past_horizon <= 0 else max(0, t - cfg.past_horizon)
        past_pts = pts[past_start : t + 1]

    scale = 1 << shift

    if len(past_pts) >= 2:
        frame_rgb = _draw_polyline_with_outline(
            frame_rgb, past_pts,
            color=cfg.past_color, thickness=cfg.past_thickness,
            outline_color=cfg.past_outline_color, outline_thickness=cfg.past_outline_thickness,
            color_end=cfg.past_color_end,
            alpha=alpha, shift=shift,
        )

    if len(future_pts) >= 2:
        frame_rgb = _draw_polyline_with_outline(
            frame_rgb, future_pts,
            color=cfg.future_color, thickness=cfg.future_thickness,
            outline_color=cfg.future_outline_color, outline_thickness=cfg.future_outline_thickness,
            color_end=cfg.future_color_end,
            is_dashed=cfg.dashed_future, dash_len=cfg.dash_len, gap_len=cfg.gap_len,
            alpha=alpha, shift=shift,
        )

    if cfg.tick_marks and len(future_pts) >= 2:
        for i in range(1, len(future_pts)):
            if i % cfg.tick_every == 0:
                frame_rgb = _draw_circle_with_outline(
                    frame_rgb, future_pts[i], cfg.tick_radius * scale,
                    color=cfg.tick_color, outline_color=cfg.tick_outline_color,
                    outline_thickness=1, alpha=alpha, shift=shift,
                )

    if len(future_pts) >= 2:
        arrow_idxs = _arrow_indices(len(future_pts), cfg)
        frame_rgb = _draw_arrows_with_outline(
            frame_rgb, future_pts, arrow_idxs,
            color=cfg.arrow_color, outline_color=cfg.arrow_outline_color,
            thickness=cfg.arrow_thickness, outline_thickness=cfg.arrow_outline_thickness,
            tip_length_px=cfg.arrow_size, alpha=alpha, shift=shift,
        )

    return _draw_circle_with_outline(
        frame_rgb, pts[t], cfg.current_dot_radius * scale,
        color=cfg.current_dot_color, outline_color=cfg.current_dot_outline_color,
        outline_thickness=cfg.current_dot_outline_thickness, alpha=alpha, shift=shift,
    )


def add_trace_overlay(
    image: Union[str, np.ndarray, Image.Image],
    points: Sequence[Tuple[float, float]],
    current_index: int = 0,
    config: TraceOverlayConfig | None = None,
    num_interpolated: int = 0,
    smoothing: float = 0.0,
) -> Image.Image:
    """Draw a trace overlay on *image* showing the trajectory through *points*.

    Parameters
    ----------
    image : str | np.ndarray | PIL.Image.Image
        The background image.
    points : sequence of (x, y) tuples
        Trajectory coordinates in pixel space.
    current_index : int
        Index into *points* representing the current position.  The yellow
        dot is drawn here, and the future trajectory extends forward.
        During training, this was set to the current frame index ``t``.
    config : TraceOverlayConfig | None
        Visual parameters. ``None`` uses training defaults (red-to-pink
        gradient, black outline, yellow dot).
    num_interpolated : int
        Number of sample points along a fitted spline.  Training used 0
        (no interpolation — raw trajectory points).
    smoothing : float
        Spline smoothing factor for ``scipy.interpolate.splprep``.

    Returns
    -------
    PIL.Image.Image
        A copy of the input image with the trace overlay drawn on top.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.copy().convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if config is None:
        config = TraceOverlayConfig()

    pts_arr = np.asarray(points, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array of (x, y) coordinates")
    if len(pts_arr) < 2:
        return img

    # Spline-interpolate for smooth curves
    n_raw = len(pts_arr)
    # Check if all points are identical (degenerate trajectory)
    if np.allclose(pts_arr, pts_arr[0]):
        return img

    if num_interpolated > 0 and n_raw >= 2:
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        if n_raw == 2:
            t_fine = np.linspace(0, 1, num_interpolated)
            xs = x[0] + (x[1] - x[0]) * t_fine
            ys = y[0] + (y[1] - y[0]) * t_fine
        else:
            k = min(3, n_raw - 1)
            tck, _ = splprep([x, y], s=smoothing, k=k)
            t_fine = np.linspace(0, 1, num_interpolated)
            xs, ys = splev(t_fine, tck)
        pts_arr = np.column_stack([xs, ys])
        current_index = round(current_index / max(n_raw - 1, 1) * (len(pts_arr) - 1))

    shift = 4
    scale = 1 << shift
    fixed_pts = [(round(cx * scale), round(cy * scale)) for cx, cy in pts_arr.tolist()]

    frame_rgb = np.array(img, dtype=np.uint8).copy()
    frame_rgb = _annotate_frame(frame_rgb, fixed_pts, current_index, config, shift=shift)

    return Image.fromarray(frame_rgb)
