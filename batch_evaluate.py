#!/usr/bin/env python3
"""
Batch evaluation of the inference trajectory pipeline on DROID episodes.

Runs the full Gemini+GPT pipeline on N episodes, logs detailed results,
and classifies each as GOOD / PARTIAL / FAIL with reasons.

Usage:
    export GEMINI_API_KEY=... OPENAI_API_KEY=...
    uv run examples/droid_trajectory_overlay/batch_evaluate.py \
        --start 0 --end 50 --output-dir batch_results
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
import traceback

import av
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
import tyro

from traj_vis_utils import TraceOverlayConfig, add_trace_overlay
from trajectory_predictor import (
    encode_pil_image,
    query_target_location,
    query_target_objects,
    query_trajectory,
    rescale_trajectory,
    resize_for_api,
)


@dataclasses.dataclass
class Args:
    start: int = 0
    end: int = 50
    output_dir: str = "batch_results"
    source_repo: str = "cadene/droid_1.0.1"
    train_repo: str = "brandonyang/droid_1.0.1_trajectory_overlay"
    gpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    # Rate limiting
    delay_between_episodes: float = 1.0


def _load_droid_ids():
    from importlib.util import spec_from_file_location, module_from_spec
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "temp_drawing", "droid_trajectory", "episode_ids.py")
    # Try multiple locations
    for candidate in [
        p,
        "/home/asethi04/ROBOTICS/temp_drawing/droid_trajectory/episode_ids.py",
    ]:
        if os.path.exists(candidate):
            spec = spec_from_file_location("episode_ids", candidate)
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.DROID_EPISODE_IDS
    raise FileNotFoundError("Cannot find episode_ids.py")


def _label(img_pil, text, pos=(3, 2), fontsize=9, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=(0, 0, 0))
    draw.text(pos, text, fill=color, font=font)
    return img_pil


def _fuzzy_find_in_locs(locs, query):
    """Try exact, substring, and word-overlap matching."""
    if not query or not locs:
        return None
    q = query.lower().strip()
    # Exact
    if query in locs:
        return locs[query]
    # Substring
    for k, v in locs.items():
        kl = k.lower()
        if q in kl or kl in q:
            return v
    # Word overlap
    q_words = set(q.split())
    for k, v in locs.items():
        if q_words & set(k.lower().split()):
            return v
    return None


def run_single_episode(frame_rgb, task, args):
    """Run full inference pipeline on a single frame.

    Returns (trajectory_dict, status, details_dict).
    status is one of: "good", "partial", "fail"
    """
    pil = Image.fromarray(frame_rgb).convert("RGB")
    original_size = pil.size
    api_img = resize_for_api(pil, max_size=1024)
    api_size = api_img.size

    details = {
        "task": task,
        "image_size": list(original_size),
        "api_size": list(api_size),
    }

    # Step 1: Decompose instruction
    try:
        target = query_target_objects(task, gpt_model=args.gpt_model)
        steps = target.get("steps", [])
        details["steps"] = steps
    except Exception as e:
        details["error"] = f"decomposition: {e}"
        return None, "fail", details

    if not steps:
        details["error"] = "no steps extracted"
        return None, "fail", details

    step = steps[0]
    manip = step["manipulating_object"]
    related = step["target_related_object"]
    target_loc = step["target_location"]

    # Enrich from later steps
    extra_query = ""
    if not target_loc and len(steps) > 1:
        ns = steps[1]
        if ns.get("target_location"):
            target_loc = ns["target_location"]
            details["enriched_target_loc"] = target_loc
        if ns.get("target_related_object") and not related:
            related = ns["target_related_object"]
            extra_query = related
            details["enriched_related"] = related

    details["manip"] = manip
    details["related"] = related
    details["target_loc"] = target_loc

    # Step 2: Detect objects
    detect_queries = list(set(q for q in [manip, related, extra_query] if q.strip()))
    details["detect_queries"] = detect_queries

    try:
        locs = query_target_location(api_img, detect_queries, model_name=args.gemini_model)
    except Exception as e:
        details["error"] = f"detection: {e}"
        return None, "fail", details

    if not locs:
        details["error"] = "detection returned empty"
        return None, "fail", details

    details["detections"] = {k: list(v) for k, v in locs.items()}

    # Resolve object points
    manip_pt = _fuzzy_find_in_locs(locs, manip)
    target_pt = _fuzzy_find_in_locs(locs, related) if related else None

    if manip_pt is None and locs:
        manip_pt = next(iter(locs.values()))
        details["manip_fallback"] = True

    if manip_pt is None:
        details["error"] = f"manip '{manip}' not found at all"
        return None, "fail", details

    if target_pt is None and related:
        remaining = {k: v for k, v in locs.items() if v != manip_pt}
        if remaining:
            target_pt = next(iter(remaining.values()))
            details["target_fallback"] = True

    if target_pt is None:
        target_pt = manip_pt
        details["target_same_as_manip"] = True

    details["manip_pt"] = list(manip_pt)
    details["target_pt"] = list(target_pt)
    details["manip_found"] = not details.get("manip_fallback", False)
    details["target_found"] = not details.get("target_same_as_manip", False)

    # Step 3: Generate trajectory
    try:
        img_encoded = encode_pil_image(api_img)
        traj = query_trajectory(
            img=api_img,
            img_encoded=img_encoded,
            task=step["step"],
            manipulating_object=manip,
            manipulating_object_point=manip_pt,
            target_related_object=related or "target area",
            target_related_object_point=target_pt,
            target_location=target_loc,
            gpt_model=args.gpt_model,
            full_task=task,
        )
        traj = rescale_trajectory(traj, api_size, original_size)
    except Exception as e:
        details["error"] = f"trajectory: {e}"
        return None, "fail", details

    pts = traj.get("trajectory", [])
    if len(pts) < 2:
        details["error"] = f"trajectory too short: {len(pts)} pts"
        return None, "fail", details

    # Compute metrics
    disp = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
    img_w = original_size[0]
    disp_pct = disp / img_w * 100

    details["n_pts"] = len(pts)
    details["displacement_px"] = round(disp, 1)
    details["displacement_pct"] = round(disp_pct, 1)
    details["reasoning"] = traj.get("reasoning", "")
    details["start_pt"] = pts[0]
    details["end_pt"] = pts[-1]

    # Classify quality
    is_press_task = any(w in task.lower() for w in [
        "press", "push button", "flip switch", "flip the switch",
        "turn on", "turn off", "switch on", "switch off",
    ])
    # Single-object tasks (open/close/move X) legitimately have no related object
    is_single_object_task = not step.get("target_related_object", "").strip()

    if is_press_task:
        if disp_pct < 1.0:
            status = "partial"
            details["issue"] = "press trajectory too small to be visible"
        else:
            status = "good"
    elif disp_pct < 3.0:
        status = "partial"
        details["issue"] = f"displacement too small ({disp_pct:.1f}%)"
    elif details.get("target_same_as_manip") and not is_single_object_task:
        # Only flag as partial if we EXPECTED to find a related object but didn't
        if disp_pct < 8.0:
            status = "partial"
            details["issue"] = "target object not detected, small displacement"
        else:
            status = "good"  # displaced enough that direction is probably reasonable
    else:
        status = "good"

    return traj, status, details


def main(args: Args):
    droid_ids = _load_droid_ids()

    ep_meta_path = hf_hub_download(args.train_repo, "meta/episodes.jsonl", repo_type="dataset")
    with open(ep_meta_path) as f:
        all_eps = [json.loads(l) for l in f]

    cfg = TraceOverlayConfig()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    results = []
    counts = {"good": 0, "partial": 0, "fail": 0, "skip": 0}

    for train_ep_idx in range(args.start, min(args.end, len(all_eps), len(droid_ids))):
        droid_ep_id = droid_ids[train_ep_idx]
        task = all_eps[train_ep_idx]["tasks"][0]
        droid_chunk = droid_ep_id // 1000

        print(f"\n[{train_ep_idx:3d}] DROID {droid_ep_id}: \"{task}\"")

        # Load frame
        try:
            vid = hf_hub_download(
                args.source_repo,
                f"videos/chunk-{droid_chunk:03d}/observation.images.exterior_1_left/episode_{droid_ep_id:06d}.mp4",
                repo_type="dataset",
            )
            container = av.open(vid)
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            container.close()
        except Exception as e:
            print(f"  SKIP: video load failed: {e}")
            counts["skip"] += 1
            continue

        # Run pipeline
        try:
            traj, status, details = run_single_episode(frames[0], task, args)
        except Exception as e:
            print(f"  FAIL: unexpected error: {e}")
            traceback.print_exc()
            counts["fail"] += 1
            results.append({"ep": train_ep_idx, "droid_ep": droid_ep_id, "task": task, "status": "fail", "error": str(e)})
            time.sleep(args.delay_between_episodes)
            continue

        counts[status] += 1
        details["ep"] = train_ep_idx
        details["droid_ep"] = droid_ep_id
        details["status"] = status
        results.append(details)

        disp = details.get("displacement_px", 0)
        disp_pct = details.get("displacement_pct", 0)
        n_pts = details.get("n_pts", 0)
        issue = details.get("issue", details.get("error", ""))
        icon = {"good": "OK", "partial": "~~", "fail": "XX"}[status]
        print(f"  [{icon}] {n_pts}pts, {disp:.0f}px ({disp_pct:.1f}%) {issue}")

        # Save visualization for non-good results (and every 10th good one)
        if status != "good" or train_ep_idx % 10 == 0:
            try:
                pts = traj.get("trajectory", []) if traj else []
                if len(pts) >= 2:
                    ov = add_trace_overlay(frames[0], pts, current_index=0, config=cfg, num_interpolated=100)
                    raw = Image.fromarray(frames[0])
                    # Side by side
                    combo = Image.new("RGB", (640 + 10, 180), (20, 20, 20))
                    combo.paste(raw, (0, 0))
                    combo.paste(ov, (330, 0))
                    combo = _label(combo, f"[{icon}] Ep {train_ep_idx}: {task}", (3, 1), fontsize=8)
                    combo = _label(combo, f"{n_pts}pts {disp:.0f}px ({disp_pct:.1f}%) {issue}", (3, 168), fontsize=7, color=(200, 200, 200))
                    combo.save(os.path.join(args.output_dir, "images", f"ep{train_ep_idx:03d}_{status}.png"))
            except Exception:
                pass

        time.sleep(args.delay_between_episodes)

    # Save all results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    total = sum(counts.values())
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {total} episodes evaluated")
    print(f"  GOOD:    {counts['good']:3d} ({counts['good']/max(total,1)*100:.0f}%)")
    print(f"  PARTIAL: {counts['partial']:3d} ({counts['partial']/max(total,1)*100:.0f}%)")
    print(f"  FAIL:    {counts['fail']:3d} ({counts['fail']/max(total,1)*100:.0f}%)")
    print(f"  SKIP:    {counts['skip']:3d} ({counts['skip']/max(total,1)*100:.0f}%)")
    print(f"{'=' * 80}")

    # Failure analysis
    fail_reasons = {}
    partial_reasons = {}
    for r in results:
        if r.get("status") == "fail":
            reason = r.get("error", "unknown")
            # Categorize
            if "detection" in reason.lower():
                cat = "detection_failure"
            elif "decomposition" in reason.lower():
                cat = "decomposition_failure"
            elif "trajectory" in reason.lower():
                cat = "trajectory_failure"
            else:
                cat = reason[:50]
            fail_reasons[cat] = fail_reasons.get(cat, 0) + 1
        elif r.get("status") == "partial":
            reason = r.get("issue", "unknown")
            partial_reasons[reason[:60]] = partial_reasons.get(reason[:60], 0) + 1

    if fail_reasons:
        print("\nFAILURE BREAKDOWN:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x {reason}")

    if partial_reasons:
        print("\nPARTIAL ISSUES:")
        for reason, count in sorted(partial_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x {reason}")

    # Save summary
    summary = {
        "total": total,
        "counts": counts,
        "fail_reasons": fail_reasons,
        "partial_reasons": partial_reasons,
        "success_rate": round((counts["good"]) / max(total - counts["skip"], 1) * 100, 1),
        "acceptable_rate": round((counts["good"] + counts["partial"]) / max(total - counts["skip"], 1) * 100, 1),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSuccess rate: {summary['success_rate']}%")
    print(f"Acceptable rate (good+partial): {summary['acceptable_rate']}%")


if __name__ == "__main__":
    main(tyro.cli(Args))
