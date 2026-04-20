"""
Real-time inference visualization for trajectory overlay pipeline.

Shows a live debug panel during robot execution with:
  - Raw exterior image (what the camera sees)
  - Annotated exterior image (what the model receives)
  - Wrist camera image
  - Text overlay with instruction, step, action info
  - Saves debug frames to disk for post-hoc analysis

Usage:
    viz = InferenceVisualizer(save_dir="debug_frames")
    viz.update(raw_ext, annotated_ext, wrist, info_dict)
    viz.close()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class InferenceInfo:
    """Current inference state for visualization."""
    instruction: str = ""
    current_step: str = ""
    step_idx: int = 0
    total_steps: int = 0
    plan_count: int = 0
    max_plans: int = 0
    action_idx: int = 0
    open_loop_horizon: int = 0
    t_step: int = 0
    trajectory_points: int = 0
    inference_time_ms: float = 0.0
    fps: float = 0.0


class InferenceVisualizer:
    """Real-time visualization of the trajectory overlay inference pipeline."""

    WINDOW_NAME = "Trajectory Overlay Inference"

    def __init__(
        self,
        save_dir: str | None = None,
        display_width: int = 320,
        show_window: bool = True,
        save_every_n: int = 1,
    ):
        """
        Args:
            save_dir: Directory to save debug frames. None to disable saving.
            display_width: Width of each image panel in the display.
            show_window: Whether to show a live OpenCV window.
            save_every_n: Save a debug frame every N steps (1 = every step).
        """
        self.save_dir = save_dir
        self.display_width = display_width
        self.show_window = show_window
        self.save_every_n = save_every_n
        self._frame_count = 0
        self._last_time = time.time()
        self._fps_history: list[float] = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._frames_dir = os.path.join(save_dir, "debug_frames")
            self._raw_dir = os.path.join(save_dir, "raw_frames")
            self._annotated_dir = os.path.join(save_dir, "annotated_frames")
            for d in [self._frames_dir, self._raw_dir, self._annotated_dir]:
                os.makedirs(d, exist_ok=True)

    def _get_font(self, size=12):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except (OSError, IOError):
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (OSError, IOError):
                return ImageFont.load_default()

    def _resize_for_display(self, img_rgb: np.ndarray) -> np.ndarray:
        """Resize image to display width, preserving aspect ratio."""
        h, w = img_rgb.shape[:2]
        scale = self.display_width / w
        new_h = int(h * scale)
        return cv2.resize(img_rgb, (self.display_width, new_h))

    def _draw_info_panel(self, width: int, info: InferenceInfo) -> np.ndarray:
        """Create the text info panel below the images."""
        panel_h = 110
        panel = Image.new("RGB", (width, panel_h), (25, 25, 35))
        draw = ImageDraw.Draw(panel)
        font = self._get_font(12)
        font_small = self._get_font(10)

        y = 5
        line_h = 18

        # Instruction (yellow)
        instr_text = info.instruction if len(info.instruction) < 80 else info.instruction[:77] + "..."
        draw.text((8, y), f"Instruction: {instr_text}", fill=(255, 255, 100), font=font)
        y += line_h

        # Current step (cyan)
        step_text = info.current_step if len(info.current_step) < 70 else info.current_step[:67] + "..."
        draw.text((8, y), f"Step {info.step_idx + 1}/{info.total_steps}: {step_text}",
                  fill=(100, 255, 255), font=font)
        y += line_h

        # Planning info (green)
        draw.text((8, y),
                  f"Plan: {info.plan_count}/{info.max_plans}  |  "
                  f"Action: {info.action_idx}/{info.open_loop_horizon}  |  "
                  f"Trajectory pts: {info.trajectory_points}",
                  fill=(100, 255, 100), font=font)
        y += line_h

        # Timing info (gray)
        draw.text((8, y),
                  f"t_step: {info.t_step}  |  "
                  f"Inference: {info.inference_time_ms:.0f}ms  |  "
                  f"FPS: {info.fps:.1f}",
                  fill=(180, 180, 180), font=font_small)
        y += line_h

        # Config reminder (dim) — matches TraceOverlayConfig() defaults
        draw.text((8, y),
                  "Config: red->pink gradient t=3, black outline=5, yellow dot r=5 [TRAINING MATCH]",
                  fill=(120, 120, 120), font=font_small)

        return np.array(panel)

    def _draw_image_labels(
        self,
        img_rgb: np.ndarray,
        label: str,
        sublabel: str = "",
        label_color: tuple = (255, 255, 255),
        border_color: tuple | None = None,
    ) -> np.ndarray:
        """Add a label bar on top of an image."""
        h, w = img_rgb.shape[:2]
        label_h = 22
        canvas = np.full((h + label_h, w, 3), 30, dtype=np.uint8)
        canvas[label_h:, :] = img_rgb

        if border_color is not None:
            # 2px border around the image
            canvas[label_h, :] = border_color
            canvas[-1, :] = border_color
            canvas[label_h:, 0] = border_color
            canvas[label_h:, -1] = border_color

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        font = self._get_font(11)
        draw.text((4, 3), label, fill=label_color, font=font)
        if sublabel:
            sw = draw.textlength(label, font=font)
            draw.text((int(sw) + 12, 3), sublabel, fill=(150, 150, 150), font=self._get_font(10))

        return np.array(pil)

    def update(
        self,
        raw_exterior: np.ndarray,
        annotated_exterior: np.ndarray,
        wrist_image: np.ndarray,
        info: InferenceInfo,
        model_input_ext: np.ndarray | None = None,
        model_input_wrist: np.ndarray | None = None,
    ) -> int:
        """Update the visualization with new frame data.

        Args:
            raw_exterior: Raw exterior camera image (RGB, any size)
            annotated_exterior: Exterior with trajectory overlay (RGB, any size)
            wrist_image: Wrist camera image (RGB, any size)
            info: Current inference state
            model_input_ext: The actual 224x224 image sent to model (optional)
            model_input_wrist: The actual 224x224 wrist image sent to model (optional)

        Returns:
            Key code from cv2.waitKey (for handling quit), or -1.
        """
        # Update FPS tracking
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            info.fps = sum(self._fps_history) / len(self._fps_history)

        # Resize images for display
        raw_disp = self._resize_for_display(raw_exterior)
        ann_disp = self._resize_for_display(annotated_exterior)
        wrist_disp = self._resize_for_display(wrist_image)

        # Ensure same height
        target_h = max(raw_disp.shape[0], ann_disp.shape[0], wrist_disp.shape[0])
        def pad_h(img, target):
            if img.shape[0] < target:
                pad = np.full((target - img.shape[0], img.shape[1], 3), 30, dtype=np.uint8)
                return np.vstack([img, pad])
            return img

        raw_disp = pad_h(raw_disp, target_h)
        ann_disp = pad_h(ann_disp, target_h)
        wrist_disp = pad_h(wrist_disp, target_h)

        # Add labels
        raw_labeled = self._draw_image_labels(raw_disp, "Exterior (raw)", f"{raw_exterior.shape[1]}x{raw_exterior.shape[0]}")
        ann_labeled = self._draw_image_labels(ann_disp, "Exterior (annotated)", "-> model",
                                               label_color=(0, 255, 0), border_color=(0, 200, 0))
        wrist_labeled = self._draw_image_labels(wrist_disp, "Wrist camera", f"{wrist_image.shape[1]}x{wrist_image.shape[0]}")

        # Build main panel: 3 images side by side
        gap = np.full((raw_labeled.shape[0], 3, 3), 30, dtype=np.uint8)
        top_row = np.hstack([raw_labeled, gap, ann_labeled, gap, wrist_labeled])

        # If we have model inputs, show them as a second row
        if model_input_ext is not None and model_input_wrist is not None:
            mi_ext = cv2.resize(model_input_ext, (112, 112))
            mi_wrist = cv2.resize(model_input_wrist, (112, 112))
            mi_ext_labeled = self._draw_image_labels(mi_ext, "Model input: ext", "224->112",
                                                      label_color=(0, 255, 0), border_color=(0, 150, 0))
            mi_wrist_labeled = self._draw_image_labels(mi_wrist, "Model input: wrist", "224->112")

            mi_gap = np.full((mi_ext_labeled.shape[0], 3, 3), 30, dtype=np.uint8)
            mi_row = np.hstack([mi_ext_labeled, mi_gap, mi_wrist_labeled])

            # Pad model input row to match top row width
            if mi_row.shape[1] < top_row.shape[1]:
                pad_w = top_row.shape[1] - mi_row.shape[1]
                mi_row = np.hstack([mi_row, np.full((mi_row.shape[0], pad_w, 3), 30, dtype=np.uint8)])
            else:
                mi_row = mi_row[:, :top_row.shape[1]]

            sep = np.full((2, top_row.shape[1], 3), 60, dtype=np.uint8)
            top_row = np.vstack([top_row, sep, mi_row])

        # Info panel
        info_panel = self._draw_info_panel(top_row.shape[1], info)

        # Final composite
        sep = np.full((2, top_row.shape[1], 3), 80, dtype=np.uint8)
        composite = np.vstack([top_row, sep, info_panel])

        # Save debug frames
        if self.save_dir and (self._frame_count % self.save_every_n == 0):
            frame_path = os.path.join(self._frames_dir, f"debug_{self._frame_count:05d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            raw_path = os.path.join(self._raw_dir, f"raw_{self._frame_count:05d}.jpg")
            cv2.imwrite(raw_path, cv2.cvtColor(raw_exterior, cv2.COLOR_RGB2BGR))

            ann_path = os.path.join(self._annotated_dir, f"ann_{self._frame_count:05d}.jpg")
            cv2.imwrite(ann_path, cv2.cvtColor(annotated_exterior, cv2.COLOR_RGB2BGR))

        self._frame_count += 1

        # Display
        key = -1
        if self.show_window:
            bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.WINDOW_NAME, bgr)
            key = cv2.waitKey(1) & 0xFF

        return key

    def show_planning_status(self, message: str):
        """Show a planning status overlay (called during API wait times)."""
        if not self.show_window:
            return

        # Create a simple status panel
        panel = np.full((60, self.display_width * 3 + 6, 3), 25, dtype=np.uint8)
        pil = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil)
        font = self._get_font(14)
        draw.text((10, 10), message, fill=(255, 200, 0), font=font)
        draw.text((10, 35), "Querying LLM for trajectory...", fill=(150, 150, 150), font=self._get_font(11))
        panel = np.array(pil)

        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.WINDOW_NAME, bgr)
        cv2.waitKey(1)

    def close(self):
        """Clean up display window."""
        if self.show_window:
            cv2.destroyWindow(self.WINDOW_NAME)

    def save_summary(self, info: InferenceInfo):
        """Save a text summary of the run."""
        if not self.save_dir:
            return
        summary_path = os.path.join(self.save_dir, "visualization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Instruction: {info.instruction}\n")
            f.write(f"Steps completed: {info.step_idx}/{info.total_steps}\n")
            f.write(f"Total plans: {info.plan_count}/{info.max_plans}\n")
            f.write(f"Total frames: {self._frame_count}\n")
            f.write(f"Average FPS: {info.fps:.1f}\n")
            f.write(f"Debug frames saved to: {self._frames_dir}\n")
