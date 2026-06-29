# ============================================================
# Dockerfile for Multimodal AI Agent
# ============================================================
# Base: Python 3.10 slim image
# Build: docker build -t agent-api:latest .
# Run:   docker run -p 8000:8000 agent-api:latest
# ============================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first for better caching
COPY requirements.txt .

# Install dependencies with Tsinghua mirror for faster download
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]