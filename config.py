# config.py
import os


# 从环境变量读取，如果未设置则使用默认值
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# 路径配置
CHROMA_DB_PATH = "./chroma_db"

# 服务配置
HOST = "0.0.0.0"
PORT = 8000

# Ollama 配置（从环境变量读取，如果没有则用默认值）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")