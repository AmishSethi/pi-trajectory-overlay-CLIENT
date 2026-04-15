"""
Standalone trajectory prediction module.

Provides LLM-based trajectory generation for tabletop manipulation:
  1. Instruction decomposition into steps (GPT)
  2. Object detection in image (Gemini Robotics-ER)
  3. Trajectory generation from object locations (GPT)
  4. Step completion checking (Gemini Robotics-ER)

No eva framework dependencies.

Requires environment variables:
  GEMINI_API_KEY  — Google Gemini API key
  OPENAI_API_KEY  — OpenAI API key
"""

from __future__ import annotations

import base64
import io
import json
import os
import textwrap
from typing import Optional

import numpy as np
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# Clients (lazily initialized)
# ---------------------------------------------------------------------------

_google_client: genai.Client | None = None
_openai_client: OpenAI | None = None


def _get_google_client() -> genai.Client:
    global _google_client
    if _google_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required")
        _google_client = genai.Client(api_key=api_key)
    return _google_client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_TARGET_TEMPLATE = """
You are an assistant that, given a natural language instruction for a robot, identifies:
1) What is the plan for the robot to complete the task?
2) For each step in the plan, identify the object that the robot must manipulate and the target location for that object.

Rules:
- "manipulating_object" must be only the physical item the robot grasps or moves, not the target location or container.
- "target_location" must describe where that object should be placed, applied, or moved (including relations like "to the left of the banana" or "onto the blue plate").
- If multiple objects are mentioned, choose the single primary object that is directly manipulated.
- If no clear target location is given, use the empty string "" for "target_location".
- Notice that you are a robot and can only manipulate one object at a time. Do NOT make a plan that needs to manipulate "stacked objects".

Example 1
Caption: Add sprinkles to the Coke in the designated container.
Reasoning: The robot shall pick up the sprinkles and add them to the Coke in the designated container.
Output:
{{
  "steps": [
    {{
      "step": "Add sprinkles to the Coke in the designated container.",
      "manipulating_object": "sprinkles",
      "target_location": "Coke in the designated container",
      "target_related_object": "Coke"
    }}
  ]
}}

Example 2
Caption: Place the pineapple to the left of the banana, ensuring they are positioned neatly.
Reasoning: The robot shall pick up the pineapple and place it to the left of the banana.
Output:
{{
  "steps": [
    {{
      "step": "Place the pineapple to the left of the banana, ensuring they are positioned neatly.",
      "manipulating_object": "pineapple",
      "target_location": "left of the banana",
      "target_related_object": "banana"
    }}
  ]
}}

Example 3
Caption: Place the toy cat to the other side of the pineapple, ensuring they are positioned neatly.
Reasoning:
Currently the toy cat is on the left side of the pineapple,
so the robot shall pick up the toy cat and place it to the right side of the pineapple.
Output:
{{
  "steps": [
    {{
      "step": "Place the toy cat to the right side of the pineapple.",
      "manipulating_object": "toy cat",
      "target_location": "left of the pineapple",
      "target_related_object": "pineapple"
    }}
  ]
}}


Example 4
Caption: Stack the cup from top to bottom of red, green, blue.
Reasoning: This is a stacking task, and the robot shall only manipulate one cup at a time.
Aviod to manipulate "stacked objects".
Therefore, we construct the stack from bottom to top.
First, The robot shall pick up the green cup firstand stack it to the blue cup as the first step.
Then, the robot shall pick up the red cup and stack it to the green cup as the second step.

Output:
{{
  "steps": [
    {{
      "step": "Move the pineapple to the left.",
      "manipulating_object": "pineapple",
      "target_location": "left of the pineapple",
      "target_related_object": "pineapple"
    }}
  ]
}}

Now process this caption:
Caption: {caption}

Return ONLY a single valid JSON object (no extra text). Use this format:

{{
  "steps": [
    {{
      "step": "...",
      "manipulating_object": "...",
      "target_location": "..."
      "target_related_object": "..."
    }},
    {{
      "step": "...",
      "manipulating_object": "...",
      "target_location": "..."
      "target_related_object": "..."
    }},
    ...
  ]
}}
"""

_PROMPT_TARGET_SCHEMA = {
    "name": "robot_plan_steps",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Natural language description of the step.",
                        },
                        "manipulating_object": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Single primary physical object the robot manipulates.",
                        },
                        "target_location": {
                            "type": "string",
                            "description": "Where the manipulating_object should be moved/placed/applied.",
                        },
                        "target_related_object": {
                            "type": "string",
                            "description": "The object used as a spatial/semantic reference for target_location.",
                        },
                    },
                    "required": ["step", "manipulating_object", "target_location", "target_related_object"],
                },
            }
        },
        "required": ["steps"],
    },
}

_TRAJECTORY_PROMPT = """
You are a spatial reasoning and motion planning assistant for tabletop robot manipulation.

INPUT:
- Full task instruction: string (the complete task the robot must perform)
- Current step: string (the specific step to plan a trajectory for)
- Image: an image of the tabletop with the objects in the scene.
- Image height and width: int, int
- Target object to manipulate location: (x, y) in pixels
- Relevant object location: (x, y) in pixels

GOAL: Predict the 2D path that the manipulated object will follow on the tabletop as the robot executes this step. This trajectory will be drawn on the camera image to guide the robot.

IMPORTANT — TRAJECTORY SCALE:
- The trajectory must show MEANINGFUL displacement. A robot manipulation typically moves objects 15-30% of the image width.
- If the task says "remove from X" or "pick up from X", the object must move AWAY from X by a substantial distance (at least 15% of image width).
- If the task says "put on Y" or "place in Y", the object must move TOWARD Y's location.
- If the target location is vague (e.g., "on the table"), choose a clear, open area of the table that is far from the starting position.
- NEVER generate a trajectory where the end point is within 10% of image width from the start point unless the task explicitly requires a small motion.

REQUIREMENTS:
1. The movement should be relative to the robot's perspective, not the camera.
2. Detect potential risks: falling off table, collision with obstacles.
3. Choose a safe reachable target point that satisfies the instruction while staying on the table.

TRAJECTORY RULES:
- Use 10 to 15 evenly spaced points along the path.
- First point: the manipulated object's current location.
- Final point: the predicted target location after the step is complete.
- Space intermediate points evenly so the trajectory forms a smooth curve.

Now process this task:
Full task: {full_task}
Current step: {task}
Image height: {height}
Image width: {width}
The object to manipulate is {manipulating_object} at: {manipulating_object_point}
The related object is {target_related_object} at: {target_related_object_point}
Target location: {target_location}

Return ONLY a single valid JSON object (no extra text). Use this format:
{{
  "reasoning": "...",
  "start_point": [x, y],
  "end_point": [x, y],
  "trajectory": [[x, y], [x, y], ...]
}}
"""

_TRAJECTORY_PROMPT_WITH_TARGET = """
You are a spatial reasoning and motion planning assistant for tabletop robot manipulation.

INPUT:
- Full task instruction: string (the complete task the robot must perform)
- Current step: string (the specific step to plan a trajectory for)
- Image: an image of the tabletop with the objects in the scene.
- Image height and width: int, int
- Target object to manipulate location: (x, y) in pixels (current position)
- Known end point: (x, y) in pixels (where the object should end up)

GOAL: Predict the 2D path from the object's CURRENT position to the known end point. This is a re-planning call — the end point is already decided, you just need to plan a smooth path from the current object location to that end point.

REQUIREMENTS:
1. The movement should be relative to the robot's perspective, not the camera.
2. Detect potential risks: falling off table, collision with obstacles.
3. The trajectory MUST end at or very near the given end point.

TRAJECTORY RULES:
- Use 10 to 15 evenly spaced points along the path.
- First point: the object's current location.
- Final point: the given end point.
- Space intermediate points evenly so the trajectory forms a smooth curve.

Now process this task:
Full task: {full_task}
Current step: {task}
Image height: {height}
Image width: {width}
The object to manipulate is {manipulating_object} at: {manipulating_object_point}
The related object is {target_related_object} at: {target_related_object_point}
Target location: {target_location}, which is at: {target_location_point}

Return ONLY a single valid JSON object (no extra text). Use this format:
{{
  "reasoning": "...",
  "start_point": [x, y],
  "end_point": [x, y],
  "trajectory": [[x, y], [x, y], ...]
}}
"""

_TRAJECTORY_SCHEMA = {
    "name": "tabletop_trajectory",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": "Brief explanation of safety + direction reasoning (robot-centric).",
            },
            "start_point": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Start (x, y) in pixel coordinates.",
            },
            "end_point": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Predicted safe target (x, y) in pixel coordinates.",
            },
            "trajectory": {
                "type": "array",
                "minItems": 10,
                "description": "List of 10-15 (x, y) points from start to end, evenly spaced along the path.",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "required": ["reasoning", "start_point", "end_point", "trajectory"],
    },
}

_STEP_COMPLETION_PROMPT = """
You are a spatial reasoning and motion planning assistant for tabletop manipulation.

We are currently performing this step in the task: {task}.

Based on this image, please determine if the step is complete.

Return ONLY a single valid JSON object (no extra text). Use this format:
{{
  "reasoning": "...",
  "is_complete": true/false
}}
"""

_STEP_COMPLETION_SCHEMA = {
    "name": "step_completion",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": "Brief justification based on visual evidence in the image.",
            },
            "is_complete": {
                "type": "boolean",
                "description": "True if the specified step is completed; otherwise false.",
            },
        },
        "required": ["reasoning", "is_complete"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_pil_image(img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json(text: str) -> str:
    """Strip markdown code fencing from JSON output."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            text = "\n".join(lines[i + 1 :])
            text = text.split("```")[0]
            break
    return text


def resize_for_api(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image so longest side is at most max_size."""
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return img


def _call_gemini(img: Image.Image, prompt: str, model_name: str = "gemini-robotics-er-1.5-preview") -> str:
    """Call Gemini Robotics-ER and return parsed JSON text."""
    client = _get_google_client()
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=100),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[img, prompt],
        config=config,
    )
    print(f"[Gemini] {response.text}")
    return _parse_json(response.text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_target_objects(
    instruction: str,
    gpt_model: str = "gpt-4o-mini",
    img_encoded: str | None = None,
) -> dict:
    """Decompose instruction into manipulation steps using GPT.

    Returns dict with "steps" key containing list of step dicts.
    """
    client = _get_openai_client()
    prompt = _PROMPT_TARGET_TEMPLATE.format(caption=instruction)
    content: list[dict] = [{"type": "text", "text": prompt}]
    if img_encoded is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_encoded}"},
        })
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=1000,
        response_format={"type": "json_schema", "json_schema": _PROMPT_TARGET_SCHEMA},
    )
    return json.loads(response.choices[0].message.content)


def query_target_location(
    img: Image.Image,
    queries: list[str],
    model_name: str = "gemini-robotics-er-1.5-preview",
) -> dict[str, tuple[float, float]] | None:
    """Detect object locations in image using Gemini Robotics-ER.

    Returns dict mapping object names to (x, y) pixel coordinates, or None on failure.
    """
    point_prompt = textwrap.dedent(f"""\
    Get all points matching the following objects: {', '.join(queries)}. The label
    returned should be an identifying name for the object detected.

    Note that there shall be multiple table corners in the image.

    The answer should follow the JSON format:
    [{{"point": [y_norm, x_norm], "label": "object-name"}}, ...]

    The points are in [y, x] format normalized to 0-1000.
    """)

    json_output = _call_gemini(img, point_prompt, model_name=model_name)
    width, height = img.size

    try:
        data = json.loads(json_output)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON from Gemini. Skipping.")
        return None

    object_locations = {}
    for item in data:
        name = item["label"]
        y_norm, x_norm = item["point"]
        x = (x_norm / 1000.0) * width
        y = (y_norm / 1000.0) * height
        object_locations[name] = (x, y)

    return object_locations


def query_trajectory(
    img: Image.Image,
    img_encoded: str,
    task: str,
    manipulating_object: str,
    manipulating_object_point: tuple[float, float],
    target_related_object: str,
    target_related_object_point: tuple[float, float],
    target_location: str,
    gpt_model: str = "gpt-4o-mini",
    target_location_point: tuple[float, float] | None = None,
    full_task: str = "",
) -> dict:
    """Generate a trajectory using GPT given object locations.

    Args:
        full_task: The complete original instruction (e.g. "Remove X from Y and
            put it on the table"). Passed to GPT so it understands the full
            context even when ``task`` is just one decomposed step.

    Returns dict with "trajectory", "start_point", "end_point", "reasoning" keys.
    """
    client = _get_openai_client()
    full_task = full_task or task

    if target_location_point is None:
        prompt = _TRAJECTORY_PROMPT.format(
            full_task=full_task,
            task=task,
            height=img.height,
            width=img.width,
            manipulating_object=manipulating_object,
            manipulating_object_point=manipulating_object_point,
            target_related_object=target_related_object,
            target_related_object_point=target_related_object_point,
            target_location=target_location if target_location else "a clear area on the table away from other objects",
        )
    else:
        prompt = _TRAJECTORY_PROMPT_WITH_TARGET.format(
            full_task=full_task,
            task=task,
            height=img.height,
            width=img.width,
            manipulating_object=manipulating_object,
            manipulating_object_point=manipulating_object_point,
            target_related_object=target_related_object,
            target_related_object_point=target_related_object_point,
            target_location=target_location if target_location else "a clear area on the table",
            target_location_point=target_location_point,
        )

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_encoded}"}},
            ],
        }],
        temperature=0.0,
        max_tokens=1000,
        response_format={"type": "json_schema", "json_schema": _TRAJECTORY_SCHEMA},
    )

    return json.loads(response.choices[0].message.content)


def query_step_completion(
    img_list: list[Image.Image],
    step: str,
    model_name: str = "gemini-robotics-er-1.5-preview",
    max_images: int = 3,
) -> dict:
    """Check whether the current manipulation step is complete.

    Returns dict with "is_complete" (bool) and "reasoning" (str) keys.
    """
    if len(img_list) > max_images:
        indices = np.linspace(0, len(img_list) - 2, max_images - 1, dtype=int).tolist()
        indices.append(len(img_list) - 1)
        img_list = [img_list[i] for i in indices]

    prompt = _STEP_COMPLETION_PROMPT.format(task=step)
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=100),
    )

    client = _get_google_client()
    contents = list(img_list) + [prompt]
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    print(f"[Gemini completion] {response.text}")
    json_output = _parse_json(response.text)
    return json.loads(json_output)


def rescale_trajectory(
    trajectory: dict,
    resized_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict:
    """Rescale trajectory coordinates from resized-image space to original-image space.

    Args:
        trajectory: dict with "trajectory", "start_point", "end_point" keys.
        resized_size: (width, height) of the resized image used for API calls.
        original_size: (width, height) of the original camera image.
    """
    rw, rh = resized_size
    ow, oh = original_size
    if rw == ow and rh == oh:
        return trajectory

    sx = ow / rw
    sy = oh / rh

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
