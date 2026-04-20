"""
Unified trajectory source interface for the pi0.5 trajectory overlay model.

Provides two trajectory generation methods behind a common interface:
  - GPTTrajectorySource: Gemini object detection + GPT waypoint generation
  - RetrievalWarpTrajectorySource: DROID retrieval + Gemini correspondence warping

Both output 2D pixel trajectories that get drawn using the same TraceOverlayConfig.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from trajectory_predictor import (
    encode_pil_image,
    query_target_location,
    query_target_objects,
    query_trajectory,
    rescale_trajectory,
    resize_for_api,
)


class TrajectorySource(ABC):
    """Abstract base for trajectory generation methods.

    Subclasses produce a list of (x, y) pixel coordinates in the target
    camera image's coordinate system, suitable for drawing with
    ``add_trace_overlay``.
    """

    @abstractmethod
    def predict(
        self,
        frame_rgb: np.ndarray,
        task: str,
        *,
        step_data: dict | None = None,
        full_task: str = "",
        target_location_point: list[float] | None = None,
    ) -> dict | None:
        """Generate a trajectory for the given frame and task.

        Args:
            frame_rgb: Current camera image (H, W, 3) uint8 RGB.
            task: The current step instruction.
            step_data: Decomposed step dict with manipulating_object, target_related_object, etc.
            full_task: The complete original instruction.
            target_location_point: Known endpoint for re-planning (optional).

        Returns:
            Dict with keys:
              - "trajectory": list of [x, y] pixel coordinates
              - "start_point": [x, y]
              - "end_point": [x, y]
              - "reasoning": str
              - "source": "gpt" or "retrieval"
            Or None on failure.
        """

    def decompose_instruction(self, task: str, frame_rgb: np.ndarray | None = None) -> list[dict]:
        """Decompose a task instruction into manipulation steps.

        Default implementation uses GPT. Subclasses can override.
        """
        img_encoded = None
        if frame_rgb is not None:
            pil = Image.fromarray(frame_rgb).convert("RGB")
            api_img = resize_for_api(pil, max_size=1024)
            img_encoded = encode_pil_image(api_img)

        target = query_target_objects(task, gpt_model="gpt-4o-mini", img_encoded=img_encoded)
        return target.get("steps", [])


class GPTTrajectorySource(TrajectorySource):
    """Generate trajectories using Gemini object detection + GPT waypoint planning.

    This is the simpler, faster method (~5s per plan). Produces 10-15 waypoints.
    """

    def __init__(
        self,
        gpt_model: str = "gpt-4o-mini",
        gemini_model: str = "gemini-robotics-er-1.5-preview",
    ):
        self.gpt_model = gpt_model
        self.gemini_model = gemini_model

    def predict(
        self,
        frame_rgb: np.ndarray,
        task: str,
        *,
        step_data: dict | None = None,
        full_task: str = "",
        target_location_point: list[float] | None = None,
    ) -> dict | None:
        if step_data is None:
            return None

        manip = step_data["manipulating_object"]
        related = step_data.get("target_related_object", "")
        target_loc = step_data.get("target_location", "")

        pil = Image.fromarray(frame_rgb).convert("RGB")
        original_size = pil.size
        api_img = resize_for_api(pil, max_size=1024)
        api_size = api_img.size

        # Detect objects
        queries = list(set(q for q in [manip, related] if q and q.strip()))
        if not queries:
            return None

        locs = query_target_location(api_img, queries, model_name=self.gemini_model)
        if not locs:
            return None

        from trajectory_predictor import _fuzzy_find
        manip_pt = _fuzzy_find(locs, manip)
        target_pt = _fuzzy_find(locs, related) if related else None

        if manip_pt is None and locs:
            manip_pt = next(iter(locs.values()))
        if manip_pt is None:
            return None
        if target_pt is None:
            remaining = {k: v for k, v in locs.items() if v != manip_pt}
            target_pt = next(iter(remaining.values())) if remaining else manip_pt

        # Generate trajectory
        img_encoded = encode_pil_image(api_img)
        traj = query_trajectory(
            img=api_img,
            img_encoded=img_encoded,
            task=task,
            manipulating_object=manip,
            manipulating_object_point=manip_pt,
            target_related_object=related or "target area",
            target_related_object_point=target_pt,
            target_location=target_loc,
            gpt_model=self.gpt_model,
            target_location_point=target_location_point,
            full_task=full_task,
        )

        traj = rescale_trajectory(traj, api_size, original_size)
        traj["source"] = "gpt"
        return traj


class RetrievalWarpTrajectorySource(TrajectorySource):
    """Generate trajectories by retrieving similar DROID episodes and warping.

    Pipeline:
    1. Find retrieval result JSON for the given task
    2. Load the retrieved DROID episode's SAM2 2D trajectory
    3. Use Gemini correspondence to find object matches
    4. Warp the 2D trajectory from DROID scene to current scene

    Requires preprocessed DROID 5K data and tether's gemini_geo_position.
    """

    def __init__(
        self,
        droid_5k_root: str | Path = "/home/jianih/common-data/jiani_common/DROID_5K_processed",
        retrieval_result_dir: str | Path | None = None,
        tether_root: str | Path = "/home/asethi04/ROBOTICS/tether",
        droid_trajectory_repo: str = "brandonyang/droid_1.0.1_trajectory_overlay",
        gemini_model: str = "gemini-robotics-er-1.5-preview",
    ):
        self.droid_5k_root = Path(droid_5k_root)
        self.retrieval_result_dir = Path(
            retrieval_result_dir or self.droid_5k_root / "retrieval_result" / "hungarian"
        )
        self.tether_root = Path(tether_root)
        self.droid_trajectory_repo = droid_trajectory_repo
        self.gemini_model = gemini_model

        # Add tether to path for correspondence imports
        tether_str = str(self.tether_root)
        if tether_str not in sys.path:
            sys.path.insert(0, tether_str)

        # Cache for loaded trajectories
        self._trajectory_cache: dict[int, list[tuple[float, float]]] = {}

    def find_retrieval_result(self, task: str) -> dict | None:
        """Find the best retrieval result JSON for the given task.

        Searches retrieval_result_dir for JSONs whose query instruction
        is most similar to the given task.
        """
        if not self.retrieval_result_dir.exists():
            print(f"[retrieval] Retrieval result dir not found: {self.retrieval_result_dir}")
            return None

        best_match = None
        best_score = -1.0

        for json_path in self.retrieval_result_dir.glob("*.json"):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                query_phrase = data.get("query", {}).get("phrase", "")
                # Simple word overlap similarity
                task_words = set(task.lower().split())
                query_words = set(query_phrase.lower().split())
                if not task_words or not query_words:
                    continue
                overlap = len(task_words & query_words)
                score = overlap / max(len(task_words | query_words), 1)
                if score > best_score:
                    best_score = score
                    best_match = data
                    best_match["_json_path"] = str(json_path)
            except Exception:
                continue

        if best_match and best_score > 0.3:
            print(f"[retrieval] Found match (score={best_score:.2f}): {best_match.get('query', {}).get('phrase', '')}")
            return best_match

        print(f"[retrieval] No good match for task: {task}")
        return None

    def get_top_droid_episode(self, retrieval_result: dict) -> tuple[int, str] | None:
        """Extract the top-ranked episode training index and instruction.

        Tries tier2 (scene-graph matched) first, falls back to tier1 (instruction matched).
        Returns (training_index, instruction). The training_index maps to a DROID episode
        via the episode_ids table.
        """
        # Tier 2: scene-graph matched results (best quality)
        tier2 = retrieval_result.get("tier2", {})
        results = tier2.get("results", [])
        if results:
            top = results[0]
            episode_id = top.get("episode_id", top.get("episode"))
            instruction = top.get("instruction", "")
            return int(episode_id), instruction

        # Tier 1: instruction-similarity ranked
        tier1 = retrieval_result.get("tier1", {})
        ranking = tier1.get("episode_ranking", [])
        if ranking:
            top = ranking[0]
            episode_id = top.get("episode_id", top.get("episode"))
            instruction = top.get("instruction", "")
            return int(episode_id), instruction

        return None

    def _get_droid_ids(self):
        """Load the training-index → DROID-episode-ID mapping."""
        if not hasattr(self, "_droid_ids"):
            from importlib.util import spec_from_file_location, module_from_spec
            ids_path = Path("/home/asethi04/ROBOTICS/temp_drawing/droid_trajectory/episode_ids.py")
            if ids_path.exists():
                spec = spec_from_file_location("episode_ids", str(ids_path))
                mod = module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._droid_ids = mod.DROID_EPISODE_IDS
            else:
                self._droid_ids = []
        return self._droid_ids

    def load_droid_2d_trajectory(self, training_index: int) -> list[tuple[float, float]] | None:
        """Load the 2D SAM2 trajectory for a DROID training episode.

        Args:
            training_index: Index into the training dataset (0-4505), NOT the
                raw DROID episode ID.

        Tries multiple sources:
        1. Extract from training dataset images (yellow dot detection) — most accurate
        2. Pre-extracted trajectory from droid_trajectory_sample
        """
        if training_index in self._trajectory_cache:
            return self._trajectory_cache[training_index]

        # Source 1: Extract from training dataset (yellow dot detection)
        traj = self._extract_from_training_dataset(training_index)
        if traj and len(traj) >= 2:
            self._trajectory_cache[training_index] = traj
            return traj

        # Source 2: Check droid_trajectory_sample (uses DROID episode IDs)
        droid_ids = self._get_droid_ids()
        if training_index < len(droid_ids):
            droid_ep_id = droid_ids[training_index]
            traj = self._load_from_trajectory_sample(droid_ep_id)
            if traj and len(traj) >= 2:
                self._trajectory_cache[training_index] = traj
                return traj

        print(f"[retrieval] No 2D trajectory found for training index {training_index}")
        return None

    def _extract_from_training_dataset(self, training_index: int) -> list[tuple[float, float]] | None:
        """Extract 2D trajectory from training dataset by detecting yellow dots."""
        try:
            import io
            import polars as pl
            from huggingface_hub import hf_hub_download

            chunk = training_index // 1000
            pq = hf_hub_download(
                self.droid_trajectory_repo,
                f"data/chunk-{chunk:03d}/episode_{training_index:06d}.parquet",
                repo_type="dataset",
            )
            df = pl.read_parquet(pq)
            nf = len(df)
            if nf < 2:
                return None

            traj = []
            for t in range(nf):
                img_data = df["exterior_image_1_left"][t]
                if isinstance(img_data, dict) and img_data.get("bytes"):
                    img = np.array(Image.open(io.BytesIO(img_data["bytes"])))
                    yellow = (img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] < 80)
                    if yellow.any():
                        ys, xs = np.where(yellow)
                        traj.append((float(xs.mean()), float(ys.mean())))
                    elif traj:
                        traj.append(traj[-1])
                    else:
                        traj.append((0.0, 0.0))

            if traj:
                print(f"[retrieval] Extracted 2D trajectory from training dataset: {len(traj)} points")
            return traj if len(traj) >= 2 else None

        except Exception as e:
            print(f"[retrieval] Cannot extract from training dataset: {e}")
            return None

    def _load_from_trajectory_sample(self, droid_ep_id: int) -> list[tuple[float, float]] | None:
        """Load from droid_trajectory_sample/trajectories.jsonl if available."""
        jsonl_path = Path("/home/asethi04/ROBOTICS/droid_trajectory_sample/trajectories.jsonl")
        if not jsonl_path.exists():
            return None

        points = []
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("episode_id") == droid_ep_id:
                    points.append((row["center_x"], row["center_y"]))

        if points:
            print(f"[retrieval] Loaded 2D trajectory from sample: {len(points)} points")
        return points if points else None

    def warp_2d_trajectory(
        self,
        source_trajectory: list[tuple[float, float]],
        source_frame: Image.Image,
        target_frame: Image.Image,
        source_instruction: str,
        target_instruction: str,
        source_img_size: tuple[int, int] = (320, 180),
    ) -> list[tuple[float, float]] | None:
        """Warp a 2D trajectory from source scene to target scene using Gemini correspondence.

        Uses tether's migrate_points_to_target_image for correspondence matching,
        then applies the average displacement to all trajectory points.
        """
        try:
            from gemini_geo_position import migrate_points_to_target_image
        except ImportError:
            print("[retrieval] Cannot import tether's gemini_geo_position. Check tether_root path.")
            return None

        # Use trajectory endpoints as source keypoints for correspondence
        start_pt = source_trajectory[0]
        end_pt = source_trajectory[-1]

        # Convert to normalized [y, x] 0-1000 format (what Gemini expects)
        src_w, src_h = source_img_size
        source_points = [
            {
                "label": "start_object",
                "point": [start_pt[1] / src_h * 1000, start_pt[0] / src_w * 1000],
            },
            {
                "label": "end_object",
                "point": [end_pt[1] / src_h * 1000, end_pt[0] / src_w * 1000],
            },
        ]

        # Run Gemini correspondence (retry up to 3 times for JSON parse errors)
        migrated = None
        for attempt in range(3):
            try:
                print(f"[retrieval] Running Gemini correspondence (attempt {attempt + 1}/3)...")
                migrated = migrate_points_to_target_image(
                    source_img=source_frame,
                    source_points=source_points,
                    target_img=target_frame,
                    retrieve_instruction=source_instruction,
                    query_instruction=target_instruction,
                )
                if migrated and len(migrated) >= 2:
                    break
            except Exception as e:
                print(f"[retrieval] Correspondence attempt {attempt + 1} failed: {e}")
                import time
                time.sleep(2)

        if not migrated or len(migrated) < 2:
            print("[retrieval] Gemini correspondence failed after retries")
            return None

        # Extract target points
        tgt_w, tgt_h = target_frame.size
        target_start = migrated[0].get("target_point", {})
        target_end = migrated[1].get("target_point", {})

        tgt_start_x = target_start.get("x", 0) / 1000.0 * tgt_w
        tgt_start_y = target_start.get("y", 0) / 1000.0 * tgt_h
        tgt_end_x = target_end.get("x", 0) / 1000.0 * tgt_w
        tgt_end_y = target_end.get("y", 0) / 1000.0 * tgt_h

        # Compute the affine transformation: source start/end → target start/end
        # For simplicity, use translation + scale
        src_dx = end_pt[0] - start_pt[0]
        src_dy = end_pt[1] - start_pt[1]
        tgt_dx = tgt_end_x - tgt_start_x
        tgt_dy = tgt_end_y - tgt_start_y

        src_len = max((src_dx**2 + src_dy**2)**0.5, 1e-6)
        tgt_len = max((tgt_dx**2 + tgt_dy**2)**0.5, 1e-6)
        scale = tgt_len / src_len

        # Apply: translate so source start → target start, then scale displacement
        warped = []
        for sx, sy in source_trajectory:
            # Displacement from source start
            dx = sx - start_pt[0]
            dy = sy - start_pt[1]
            # Scale and rotate to match target displacement direction
            if src_len > 1e-6:
                # Rotation angle
                src_angle = np.arctan2(src_dy, src_dx)
                tgt_angle = np.arctan2(tgt_dy, tgt_dx)
                rot = tgt_angle - src_angle
                cos_r, sin_r = np.cos(rot), np.sin(rot)
                rx = dx * cos_r - dy * sin_r
                ry = dx * sin_r + dy * cos_r
                wx = tgt_start_x + rx * scale
                wy = tgt_start_y + ry * scale
            else:
                wx = tgt_start_x + dx
                wy = tgt_start_y + dy
            warped.append((float(wx), float(wy)))

        print(f"[retrieval] Warped {len(warped)} points: "
              f"({warped[0][0]:.0f},{warped[0][1]:.0f}) → ({warped[-1][0]:.0f},{warped[-1][1]:.0f})")
        return warped

    def predict(
        self,
        frame_rgb: np.ndarray,
        task: str,
        *,
        step_data: dict | None = None,
        full_task: str = "",
        target_location_point: list[float] | None = None,
    ) -> dict | None:
        """Retrieve a similar DROID episode and warp its trajectory."""
        effective_task = full_task or task

        # 1. Find retrieval result
        retrieval = self.find_retrieval_result(effective_task)
        if retrieval is None:
            print("[retrieval] No retrieval match, falling back to GPT")
            return None

        # 2. Get top DROID episode
        top = self.get_top_droid_episode(retrieval)
        if top is None:
            print("[retrieval] No ranked episodes found")
            return None

        training_index, droid_instruction = top
        droid_ids = self._get_droid_ids()
        droid_ep_id = droid_ids[training_index] if training_index < len(droid_ids) else training_index
        print(f"[retrieval] Top episode: training_idx={training_index}, DROID ep={droid_ep_id} — \"{droid_instruction}\"")

        # 3. Load 2D trajectory (uses training index)
        source_traj = self.load_droid_2d_trajectory(training_index)
        if not source_traj or len(source_traj) < 2:
            print(f"[retrieval] No valid 2D trajectory for training index {training_index}")
            return None

        # 4. Load source frame from DROID
        try:
            import av
            from huggingface_hub import hf_hub_download
            droid_chunk = droid_ep_id // 1000
            vid_path = hf_hub_download(
                "cadene/droid_1.0.1",
                f"videos/chunk-{droid_chunk:03d}/observation.images.exterior_1_left/episode_{droid_ep_id:06d}.mp4",
                repo_type="dataset",
            )
            container = av.open(vid_path)
            source_frame = Image.fromarray(
                next(container.decode(video=0)).to_ndarray(format="rgb24")
            )
            container.close()
        except Exception as e:
            print(f"[retrieval] Cannot load source frame: {e}")
            return None

        # 5. Warp trajectory
        target_frame = Image.fromarray(frame_rgb).convert("RGB")
        warped = self.warp_2d_trajectory(
            source_trajectory=source_traj,
            source_frame=source_frame,
            target_frame=target_frame,
            source_instruction=droid_instruction,
            target_instruction=effective_task,
            source_img_size=(320, 180),  # DROID native resolution
        )

        if not warped or len(warped) < 2:
            print("[retrieval] Warping failed")
            return None

        return {
            "trajectory": warped,
            "start_point": list(warped[0]),
            "end_point": list(warped[-1]),
            "reasoning": f"Retrieved training_idx={training_index} (DROID ep {droid_ep_id}, \"{droid_instruction}\") and warped {len(source_traj)} points",
            "source": "retrieval",
            "training_index": training_index,
            "droid_episode_id": droid_ep_id,
            "droid_instruction": droid_instruction,
            "original_trajectory": source_traj,
        }


class FallbackTrajectorySource(TrajectorySource):
    """Try retrieval+warp first, fall back to GPT if retrieval fails."""

    def __init__(
        self,
        retrieval_source: RetrievalWarpTrajectorySource,
        gpt_source: GPTTrajectorySource,
    ):
        self.retrieval = retrieval_source
        self.gpt = gpt_source

    def predict(self, frame_rgb, task, **kwargs) -> dict | None:
        result = self.retrieval.predict(frame_rgb, task, **kwargs)
        if result is not None:
            return result
        print("[fallback] Retrieval failed, using GPT trajectory source")
        return self.gpt.predict(frame_rgb, task, **kwargs)
