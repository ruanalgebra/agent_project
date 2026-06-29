"""
Configuration management for the Multimodal AI Agent.

All settings are loaded from environment variables with sensible defaults.
"""

import os

# ---------- Environment ----------
ENV = os.getenv("ENV", "development")  # development | production

# ---------- Logging ----------
LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "DEBUG" if ENV == "development" else "INFO"
)

# ---------- Model Settings ----------
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3-vl:8b-instruct-q4_K_M"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

# ---------- Paths ----------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ---------- Server ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# ---------- Ollama Service ----------
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)