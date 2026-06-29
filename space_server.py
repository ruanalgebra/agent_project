"""
Spatial Understanding Service (runs on host machine)

Provides HTTP endpoints for Docker containers (or other services) to request
spatial understanding from the camera via Qwen3-VL.

Endpoints:
- GET /space  : Returns JSON with spatial analysis of the current scene.
- GET /health : Health check.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from space_understand import space_understand

# ---------- Configuration ----------
HOST = os.getenv("SPACE_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SPACE_SERVER_PORT", 8001))

# ---------- FastAPI App ----------
app = FastAPI(
    title="Spatial Understanding Service",
    version="1.0",
    description="Camera-based spatial understanding via Qwen3-VL.",
)

# Enable CORS for container access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Endpoints ----------
@app.get("/space")
async def get_space():
    """
    Capture a camera frame and return spatial understanding as JSON.

    Response format:
    {
        "objects": [...],
        "spatial": {
            "front": "...",
            "left": "...",
            "right": "...",
            "obstacles": "..."
        },
        "summary": "..."
    }
    """
    result = space_understand()
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Entry Point ----------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)