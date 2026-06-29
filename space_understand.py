"""
Spatial Understanding Module

Captures an image from the camera and uses Qwen3-VL to analyze spatial layout,
object positions, and obstacles. Outputs structured JSON for agent consumption.

Dependencies:
- OpenCV (cv2) for camera capture
- Qwen3-VL via Ollama API
"""

import json
import re
import base64
from typing import Optional, Tuple, Dict, Any

import cv2
import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ---------- Camera Capture ----------
def capture_image(max_attempts: int = 3) -> Tuple[Optional[str], Optional[str]]:
    """
    Capture a frame from the default camera and return as base64 string.

    Args:
        max_attempts: Number of camera indices to try (0, 1, 2...).

    Returns:
        Tuple of (base64_image, error_message).
        If successful, error is None; otherwise image is None.
    """
    for idx in range(max_attempts):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_b64 = base64.b64encode(img_encoded).decode('utf-8')
                return img_b64, None

    return None, "❌ Failed to open camera (tried indices 0..{max_attempts-1})."


# ---------- Scene Understanding ----------
def describe_scene_with_space(img_base64: str) -> Dict[str, Any]:
    """
    Send the image to Qwen3-VL with a prompt for spatial analysis.

    Returns:
        A dictionary containing:
        - "objects": list of detected objects
        - "spatial": dict with front/left/right/obstacles descriptions
        - "summary": overall assessment and suggestion
    """
    prompt = """
Analyze the spatial layout and object positions in this image. Return ONLY valid JSON with the following structure:

{
    "objects": ["object1", "object2", ...],
    "spatial": {
        "front": "Describe what is ahead and its distance",
        "left": "Describe what is on the left",
        "right": "Describe what is on the right",
        "obstacles": "Are there obstacles? Where and how far?"
    },
    "summary": "One-sentence summary of traversability and suggested action."
}
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

    if response.status_code != 200:
        return {"error": f"Model error: HTTP {response.status_code}"}

    content = response.json().get("response", "")
    if not content:
        return {"error": "Empty response from model."}

    # Extract JSON from the response
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if not json_match:
        return {"raw": content, "error": "No JSON found in response."}

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"raw": content, "error": f"JSON parsing failed: {e}"}


# ---------- Public API ----------
def space_understand() -> Dict[str, Any]:
    """
    Perform full pipeline: capture image → analyze with Qwen3-VL → return structured JSON.

    Returns:
        Dictionary with either spatial data or an "error" field.
    """
    img_b64, err = capture_image()
    if err:
        return {"error": err}
    return describe_scene_with_space(img_b64)


# ---------- CLI Test ----------
if __name__ == "__main__":
    print("📷 Capturing and analyzing...")
    result = space_understand()
    print(json.dumps(result, indent=2, ensure_ascii=False))