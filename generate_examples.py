#!/usr/bin/env python3
"""
Generate example trajectory overlay visualizations.

Creates a gallery showing exactly what the model sees at inference time:
  - Various trajectory patterns on synthetic tabletop scenes
  - Full resolution vs 224x224 model input comparison
  - Training-matched config vs wrong configs (to show why matching matters)
  - Side-by-side raw vs annotated images

Usage:
    cd examples/droid_trajectory_overlay
    python generate_examples.py [--output-dir example_outputs]
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from traj_vis_utils import TraceOverlayConfig, add_trace_overlay


def create_synthetic_tabletop(width=640, height=480):
    """Create a synthetic tabletop scene image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Table surface (brown-ish)
    img[100:height, :] = [90, 120, 160]  # BGR-ish brown in RGB

    # Darker table edge
    img[95:105, :] = [60, 80, 110]

    # Background wall
    img[0:100, :] = [180, 190, 200]

    # Object 1: Red cup (circle)
    cv2.circle(img, (150, 280), 35, (200, 50, 50), -1)
    cv2.circle(img, (150, 280), 35, (150, 30, 30), 2)

    # Object 2: Green box
    cv2.rectangle(img, (420, 250), (500, 320), (50, 180, 50), -1)
    cv2.rectangle(img, (420, 250), (500, 320), (30, 130, 30), 2)

    # Object 3: Blue ball
    cv2.circle(img, (320, 350), 25, (60, 60, 200), -1)
    cv2.circle(img, (320, 350), 25, (40, 40, 150), 2)

    # Object 4: Yellow star-ish thing (small)
    cv2.circle(img, (250, 200), 15, (200, 200, 50), -1)

    # Gripper shadow area (top)
    cv2.rectangle(img, (280, 110), (360, 160), [70, 90, 130], -1)

    return img


def add_text_label(img_pil, text, position=(10, 10), color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add a text label to a PIL image."""
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox(position, text, font=font)
    # Background rectangle
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg_color)
    draw.text(position, text, fill=color, font=font)
    return img_pil


def save_with_label(img, label, path):
    """Save image with a label in the top-left corner."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = add_text_label(img.copy(), label)
    img.save(path)
    print(f"  Saved: {path}")


def generate_trajectory_patterns(output_dir):
    """Generate examples showing different trajectory patterns."""
    scene = create_synthetic_tabletop()
    patterns_dir = os.path.join(output_dir, "01_trajectory_patterns")
    os.makedirs(patterns_dir, exist_ok=True)

    # Save raw scene
    save_with_label(scene, "Raw scene (no trajectory)", os.path.join(patterns_dir, "00_raw_scene.png"))

    # Pattern 1: Simple pick-and-place (cup to box)
    pts_pick_place = [(150, 280), (200, 250), (300, 230), (400, 250), (460, 280)]
    result = add_trace_overlay(scene, pts_pick_place)
    save_with_label(result, "Pick-and-place: cup -> box", os.path.join(patterns_dir, "01_pick_place.png"))

    # Pattern 2: Push to the side
    pts_push = [(320, 350), (350, 350), (400, 340), (450, 330), (500, 320)]
    result = add_trace_overlay(scene, pts_push)
    save_with_label(result, "Push: ball to the right", os.path.join(patterns_dir, "02_push_right.png"))

    # Pattern 3: Curved sweep
    pts_curve = [
        (150, 280), (180, 240), (220, 210),
        (270, 200), (320, 210), (350, 240), (360, 280),
    ]
    result = add_trace_overlay(scene, pts_curve)
    save_with_label(result, "Curved sweep trajectory", os.path.join(patterns_dir, "03_curved_sweep.png"))

    # Pattern 4: Short precise move
    pts_short = [(250, 200), (270, 195), (290, 200)]
    result = add_trace_overlay(scene, pts_short)
    save_with_label(result, "Short precise move", os.path.join(patterns_dir, "04_short_move.png"))

    # Pattern 5: Long diagonal
    pts_diagonal = [(100, 400), (200, 350), (300, 300), (400, 250), (500, 200)]
    result = add_trace_overlay(scene, pts_diagonal)
    save_with_label(result, "Long diagonal trajectory", os.path.join(patterns_dir, "05_long_diagonal.png"))

    # Pattern 6: Stacking (upward motion)
    pts_stack = [(150, 280), (150, 250), (150, 220), (150, 190)]
    result = add_trace_overlay(scene, pts_stack)
    save_with_label(result, "Stacking: upward lift", os.path.join(patterns_dir, "06_stack_up.png"))


def generate_resolution_comparison(output_dir):
    """Show full-resolution vs 224x224 model input."""
    scene = create_synthetic_tabletop()
    res_dir = os.path.join(output_dir, "02_resolution_comparison")
    os.makedirs(res_dir, exist_ok=True)

    trajectory = [(150, 280), (220, 240), (300, 220), (400, 240), (460, 280)]

    # Full resolution annotated
    full_res = add_trace_overlay(scene, trajectory)
    save_with_label(full_res, f"Full resolution ({scene.shape[1]}x{scene.shape[0]})",
                    os.path.join(res_dir, "01_full_resolution.png"))

    # What the model actually sees: 224x224
    full_arr = np.array(full_res)
    model_input = Image.fromarray(full_arr).resize((224, 224), Image.Resampling.LANCZOS)
    save_with_label(model_input, "Model input (224x224)",
                    os.path.join(res_dir, "02_model_input_224.png"))

    # Upscaled model input (to see detail)
    upscaled = model_input.resize((448, 448), Image.Resampling.NEAREST)
    save_with_label(upscaled, "Model input upscaled 2x (nearest neighbor)",
                    os.path.join(res_dir, "03_model_input_upscaled.png"))

    # Side-by-side: raw vs annotated at model resolution
    raw_224 = Image.fromarray(scene).resize((224, 224), Image.Resampling.LANCZOS)
    side_by_side = Image.new("RGB", (448 + 10, 224), (40, 40, 40))
    side_by_side.paste(raw_224, (0, 0))
    side_by_side.paste(model_input, (234, 0))
    side_by_side = add_text_label(side_by_side, "Raw", (5, 5))
    side_by_side = add_text_label(side_by_side, "With trajectory", (239, 5))
    side_by_side.save(os.path.join(res_dir, "04_side_by_side_224.png"))
    print(f"  Saved: {os.path.join(res_dir, '04_side_by_side_224.png')}")


def generate_config_comparison(output_dir):
    """Show training config vs WRONG configs to illustrate why matching matters."""
    scene = create_synthetic_tabletop()
    cfg_dir = os.path.join(output_dir, "03_config_comparison")
    os.makedirs(cfg_dir, exist_ok=True)

    trajectory = [(150, 280), (220, 240), (300, 220), (400, 240), (460, 280)]

    # 1. CORRECT: Training defaults (magenta, thin, no outline)
    correct_cfg = TraceOverlayConfig()  # defaults
    result = add_trace_overlay(scene, trajectory, config=correct_cfg)
    save_with_label(result, "CORRECT: Training config (magenta, t=1, no outline)",
                    os.path.join(cfg_dir, "01_correct_training_config.png"))

    # 2. WRONG: Red-to-pink gradient (what add_arrow uses)
    wrong_cfg1 = TraceOverlayConfig(
        future_color=(255, 0, 0),
        future_color_end=(255, 105, 180),
        future_thickness=3,
        future_outline_thickness=6,
        future_outline_color=(0, 0, 0),
    )
    result = add_trace_overlay(scene, trajectory, config=wrong_cfg1)
    save_with_label(result, "WRONG: add_arrow config (red->pink, thick, outlined)",
                    os.path.join(cfg_dir, "02_wrong_add_arrow.png"))

    # 3. WRONG: Too thick
    wrong_cfg2 = TraceOverlayConfig(
        future_color=(255, 0, 255),
        future_thickness=5,
    )
    result = add_trace_overlay(scene, trajectory, config=wrong_cfg2)
    save_with_label(result, "WRONG: Too thick (t=5)",
                    os.path.join(cfg_dir, "03_wrong_too_thick.png"))

    # 4. WRONG: With arrows and dots
    wrong_cfg3 = TraceOverlayConfig(
        future_color=(255, 0, 255),
        future_thickness=1,
        arrow_count=1,
        arrow_size=14,
        arrow_color=(0, 255, 255),
        current_dot_radius=6,
        current_dot_color=(255, 255, 0),
    )
    result = add_trace_overlay(scene, trajectory, config=wrong_cfg3)
    save_with_label(result, "WRONG: With arrows and dot",
                    os.path.join(cfg_dir, "04_wrong_arrows_dots.png"))

    # 5. WRONG: Wrong color (green)
    wrong_cfg4 = TraceOverlayConfig(
        future_color=(0, 255, 0),
        future_thickness=1,
    )
    result = add_trace_overlay(scene, trajectory, config=wrong_cfg4)
    save_with_label(result, "WRONG: Green instead of magenta",
                    os.path.join(cfg_dir, "05_wrong_green.png"))

    # 6. All at model resolution (224x224) for direct comparison
    configs = [
        ("Correct (training)", correct_cfg),
        ("Wrong (add_arrow)", wrong_cfg1),
        ("Wrong (thick)", wrong_cfg2),
        ("Wrong (arrows)", wrong_cfg3),
    ]
    tiles = []
    for label, cfg in configs:
        result = add_trace_overlay(scene, trajectory, config=cfg)
        tile = np.array(result.resize((224, 224), Image.Resampling.LANCZOS))
        # Add label
        tile_pil = add_text_label(Image.fromarray(tile), label, (3, 3))
        tiles.append(np.array(tile_pil))

    grid = np.concatenate([
        np.concatenate(tiles[:2], axis=1),
        np.concatenate(tiles[2:], axis=1),
    ], axis=0)
    Image.fromarray(grid).save(os.path.join(cfg_dir, "06_grid_comparison_224.png"))
    print(f"  Saved: {os.path.join(cfg_dir, '06_grid_comparison_224.png')}")


def generate_inference_debug_view(output_dir):
    """Generate an example of the debug view shown during inference."""
    scene = create_synthetic_tabletop()
    debug_dir = os.path.join(output_dir, "04_inference_debug_view")
    os.makedirs(debug_dir, exist_ok=True)

    trajectory = [(150, 280), (220, 240), (300, 220), (400, 240), (460, 280)]

    # Simulate what the debug view shows during inference
    # Left: raw exterior image
    # Middle: annotated exterior image (what model sees)
    # Right: wrist image
    raw_ext = scene.copy()
    annotated_ext = np.array(add_trace_overlay(scene, trajectory))
    wrist = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    # Add a "gripper" shape to wrist image
    cv2.rectangle(wrist, (250, 200), (390, 350), (100, 100, 100), -1)

    # Resize all to same height for side-by-side
    h = 240
    w = int(640 * h / 480)
    raw_resized = cv2.resize(raw_ext, (w, h))
    ann_resized = cv2.resize(annotated_ext, (w, h))
    wrist_resized = cv2.resize(wrist, (w, h))

    # Create debug panel
    gap = 5
    panel_w = w * 3 + gap * 2
    info_h = 80
    panel = np.full((h + info_h, panel_w, 3), 30, dtype=np.uint8)

    # Place images
    panel[0:h, 0:w] = raw_resized
    panel[0:h, w + gap:2 * w + gap] = ann_resized
    panel[0:h, 2 * w + 2 * gap:3 * w + 2 * gap] = wrist_resized

    panel_pil = Image.fromarray(panel)

    # Add labels
    panel_pil = add_text_label(panel_pil, "Exterior (raw)", (5, 5))
    panel_pil = add_text_label(panel_pil, "Exterior (annotated -> model)", (w + gap + 5, 5), color=(0, 255, 0))
    panel_pil = add_text_label(panel_pil, "Wrist camera", (2 * w + 2 * gap + 5, 5))

    # Add info bar
    info_y = h + 5
    panel_pil = add_text_label(panel_pil, "Instruction: Pick up the red cup and place it on the green box",
                               (10, info_y), color=(255, 255, 100))
    panel_pil = add_text_label(panel_pil, "Step 1/1: Pick up the red cup and place on green box  |  "
                               "Plan 3/20  |  Action chunk 5/8  |  t=47",
                               (10, info_y + 20), color=(200, 200, 200))
    panel_pil = add_text_label(panel_pil, "Trajectory: 5 pts, from (150,280) to (460,280)  |  "
                               "Config: magenta t=1 (training match)",
                               (10, info_y + 40), color=(180, 180, 180))

    panel_pil.save(os.path.join(debug_dir, "inference_debug_view.png"))
    print(f"  Saved: {os.path.join(debug_dir, 'inference_debug_view.png')}")

    # Also save the model-input view (what actually goes to the model)
    model_view_w = 224
    ext_224 = Image.fromarray(annotated_ext).resize((model_view_w, model_view_w), Image.Resampling.LANCZOS)
    wrist_224 = Image.fromarray(wrist).resize((model_view_w, model_view_w), Image.Resampling.LANCZOS)

    model_panel = Image.new("RGB", (model_view_w * 2 + 10, model_view_w + 30), (30, 30, 30))
    model_panel.paste(ext_224, (0, 25))
    model_panel.paste(wrist_224, (model_view_w + 10, 25))
    model_panel = add_text_label(model_panel, "exterior_image_1_left (224x224)", (5, 5), color=(0, 255, 0))
    model_panel = add_text_label(model_panel, "wrist_image_left (224x224)", (model_view_w + 15, 5))

    model_panel.save(os.path.join(debug_dir, "model_input_actual.png"))
    print(f"  Saved: {os.path.join(debug_dir, 'model_input_actual.png')}")


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory overlay example visualizations")
    parser.add_argument("--output-dir", default="example_outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating examples in {args.output_dir}/\n")

    print("1. Trajectory patterns...")
    generate_trajectory_patterns(args.output_dir)

    print("\n2. Resolution comparison...")
    generate_resolution_comparison(args.output_dir)

    print("\n3. Config comparison (correct vs wrong)...")
    generate_config_comparison(args.output_dir)

    print("\n4. Inference debug view...")
    generate_inference_debug_view(args.output_dir)

    print(f"\nDone! All examples saved to {args.output_dir}/")
    print(f"  01_trajectory_patterns/  — Various trajectory shapes")
    print(f"  02_resolution_comparison/ — Full-res vs 224x224 model input")
    print(f"  03_config_comparison/     — Training config vs wrong configs")
    print(f"  04_inference_debug_view/  — What you see during live inference")


if __name__ == "__main__":
    main()
