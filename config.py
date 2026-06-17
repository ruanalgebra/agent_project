# config.py
# 模型配置
OLLAMA_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
EMBEDDING_MODEL = "nomic-embed-text"

# 路径配置
CHROMA_DB_PATH = "./chroma_db"

# 服务配置
HOST = "0.0.0.0"
PORT = 8000

# Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434"