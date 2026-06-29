"""
Multimodal AI Agent API - Main Application

A LangChain-based agent with tools for time, math, image description,
RAG retrieval, and spatial understanding (camera + Qwen3-VL).
"""

import os
import sys
import time
import uuid
import base64
from datetime import datetime
from collections import defaultdict

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    HOST,
    PORT,
    OLLAMA_BASE_URL,
    LOG_LEVEL,
)
from space_understand import space_understand

# ---------- Logging Setup ----------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[request_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[request_id]} | {name}:{function}:{line} | {message}",
    level=LOG_LEVEL,
)

print(f"🔍 Agent using model: {OLLAMA_MODEL}")

# ---------- FastAPI App ----------
app = FastAPI(title="Multimodal AI Agent API", version="1.0")

# ---------- Tools ----------
@tool
def get_current_time() -> str:
    """Return current date and time in natural Chinese expression."""
    now = datetime.now()
    hour_12 = now.hour % 12 or 12
    if 5 <= now.hour < 8:
        am_pm = "早上"
    elif 8 <= now.hour < 12:
        am_pm = "上午"
    elif 12 <= now.hour < 18:
        am_pm = "下午"
    else:
        am_pm = "晚上"
    return f"{now.year}年{now.month}月{now.day}日 {am_pm}{hour_12}点{now.minute}分{now.second}秒"


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def describe_image(image_path: str) -> str:
    """Describe an image. Supports relative paths."""
    abs_path = os.path.abspath(image_path)
    logger.debug(f"[describe_image] Path: {abs_path}, exists: {os.path.exists(abs_path)}")
    if not os.path.exists(abs_path):
        return f"❌ Image file not found: {abs_path}"

    try:
        with open(abs_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"❌ Failed to read image: {e}"

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Please describe this image in detail.",
                "images": [img_base64],
                "stream": False,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "No description generated.")
        return f"❌ Model error: {resp.status_code}"
    except requests.exceptions.Timeout:
        return "❌ Inference timeout."
    except Exception as e:
        return f"❌ Request failed: {e}"


# ---------- RAG Setup ----------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


@tool
def query_knowledge(question: str) -> str:
    """Retrieve relevant information from local knowledge base."""
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    if not docs:
        return "No relevant information found in knowledge base."
    content = "\n\n".join([d.page_content for d in docs])
    prompt = f"Based on the following information, answer the question:\n\n{content}\n\nQuestion: {question}\nAnswer: "
    return llm.invoke(prompt).content


@tool
def look_around() -> str:
    """Observe surroundings via camera and return spatial understanding."""
    try:
        resp = requests.get("http://localhost:8001/space", timeout=15)
        if resp.status_code != 200:
            return f"❌ Spatial service error: HTTP {resp.status_code}"

        data = resp.json()
        if "error" in data:
            return f"❌ Spatial understanding failed: {data['error']}"

        spatial = data.get("spatial", {})
        summary = data.get("summary", "")
        return f"""
📷 Scene Analysis:
- Front: {spatial.get('front', 'unknown')}
- Left: {spatial.get('left', 'unknown')}
- Right: {spatial.get('right', 'unknown')}
- Obstacles: {spatial.get('obstacles', 'none')}
💡 Suggestion: {summary}
"""
    except requests.exceptions.ConnectionError:
        return "❌ Cannot reach spatial service. Ensure space_server.py is running on host (port 8001)."
    except requests.exceptions.Timeout:
        return "❌ Spatial service timeout. Check camera."
    except Exception as e:
        return f"❌ Spatial service error: {e}"


# ---------- Agent Initialization ----------
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
tools = [get_current_time, add_numbers, describe_image, query_knowledge, look_around]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """
You are a helpful assistant that can call tools to answer questions.

Rules:
- When user asks "look around", "observe surroundings", "what's around me", "看看周围", "观察环境" or similar, call `look_around`.
- When user mentions "Ouyang Chao", "learning path", "expertise", "欧阳超", "学习路线", "擅长", call `query_knowledge`.
- If `query_knowledge` returns nothing, you may answer from your own knowledge.
- Respond in the same language as the user's input.
- For other cases, call the appropriate tool.
Keep responses concise and accurate.
"""

agent = create_agent(
    model=llm_with_tools,
    tools=tools,
    system_prompt=system_prompt,
)

# ---------- API Models ----------
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    code: int
    data: str
    msg: str = "success"


# In-memory session storage (replace with Redis for production)
session_memory = defaultdict(list)

# ---------- Endpoints ----------
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    request_id = str(uuid.uuid4())[:8]
    with logger.contextualize(request_id=request_id):
        start = time.time()
        logger.info(f"Request | session={request.session_id} | q='{request.question[:50]}...'")
        try:
            msgs = session_memory[request.session_id]
            msgs.append(("user", request.question))
            result = agent.invoke({"messages": msgs})
            reply = result["messages"][-1].content
            msgs.append(("assistant", reply))
            session_memory[request.session_id] = msgs

            # Extract token usage if available
            usage = {}
            meta = result["messages"][-1].response_metadata
            if meta:
                if "prompt_eval_count" in meta and "eval_count" in meta:
                    usage = {
                        "prompt_tokens": meta.get("prompt_eval_count", 0),
                        "completion_tokens": meta.get("eval_count", 0),
                        "total_tokens": meta.get("prompt_eval_count", 0) + meta.get("eval_count", 0),
                    }

            elapsed = time.time() - start
            logger.info(
                f"Done | time={elapsed:.3f}s | len={len(reply)} | "
                f"prompt={usage.get('prompt_tokens', 'N/A')} | "
                f"completion={usage.get('completion_tokens', 'N/A')} | "
                f"total={usage.get('total_tokens', 'N/A')}"
            )
            return QueryResponse(code=200, data=reply)
        except Exception as e:
            logger.error(f"Failed | time={time.time()-start:.3f}s | error={e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "ollama": {"reachable": False, "model_loaded": False},
        "chromadb": {"available": False},
    }

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            status["ollama"]["reachable"] = True
            for m in resp.json().get("models", []):
                if m.get("name") == OLLAMA_MODEL:
                    status["ollama"]["model_loaded"] = True
                    break
    except:
        status["status"] = "degraded"

    # Check ChromaDB
    if os.path.exists(CHROMA_DB_PATH):
        status["chromadb"]["available"] = True
    else:
        status["status"] = "degraded"

    return status


# ---------- Entry Point ----------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)