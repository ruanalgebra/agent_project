from config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    HOST,
    PORT,
    OLLAMA_BASE_URL
)
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from datetime import datetime
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import requests
import base64
import os
import uvicorn
import uuid
import time

# ---------- 日志配置 ----------
from loguru import logger
import sys
from config import LOG_LEVEL

# 移除默认 handler，自定义格式
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[request_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL # 使用变量
)
# 同时写入文件（可选）
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[request_id]} | {name}:{function}:{line} | {message}",
    level=LOG_LEVEL # 使用变量
)


# ---------- 1. 初始化 FastAPI ----------
app = FastAPI(title="多模态 AI Agent API", version="1.0")

# ---------- 2. 定义工具（直接从你 first_agent.py 复制过来，稍微润色）----------
@tool
def get_current_time() -> str:
    """获取当前日期和时间，返回中文自然表达"""
    now = datetime.now()
    hour_12 = now.hour % 12
    if hour_12 == 0:
        hour_12 = 12
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
    """计算两个数字的和"""
    return a + b

@tool
def describe_image(image_path: str) -> str:
    """分析图片内容并返回描述。支持相对路径。"""
    # 使用 logger 替代 print，但不带 request_id（工具内部无法获取），保留基础日志
    logger.debug(f"[describe_image] 收到原始路径: {image_path}")
    logger.debug(f"[describe_image] repr: {repr(image_path)}")

    abs_path = os.path.abspath(image_path)
    logger.debug(f"[describe_image] 绝对路径: {abs_path}")
    logger.debug(f"[describe_image] 文件是否存在: {os.path.exists(abs_path)}")
    
    if not os.path.exists(abs_path):
        return f"❌ 图片文件不存在：{abs_path}。请检查路径是否正确。"
    try:
        with open(abs_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"❌ 读取图片失败：{str(e)}"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "请详细描述这张图片的内容。",
                "images": [img_base64],
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "无法获取描述")
        else:
            return f"❌ 模型调用失败，状态码: {response.status_code}"
    except requests.exceptions.Timeout:
        return "❌ 模型推理超时，请稍后再试。"
    except Exception as e:
        return f"❌ 请求出错：{str(e)}"

# ---------- 3. 加载知识库 (RAG) ----------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url = OLLAMA_BASE_URL)
# 注意：如果你的 chroma_db 路径变了，记得改这里
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

@tool
def query_knowledge(question: str) -> str:
    """从本地知识库中检索相关信息"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    if not docs:
        return "知识库中没有找到相关信息。"
    content = "\n\n".join([doc.page_content for doc in docs])
    # 这里直接用 llm 生成简洁答案（注意避免循环依赖，把 llm 定义放前面）
    prompt = f"根据以下信息回答问题: \n\n{content}\n\n问题: {question}\n答案: "
    respon = llm.invoke(prompt)
    return respon.content

# ---------- 4. 初始化 LLM 和 Agent ----------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    base_url = OLLAMA_BASE_URL
)

tools = [get_current_time, add_numbers, describe_image, query_knowledge]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """
你是一个有帮助的助手，可以调用工具来回答问题。
**强制规则**：
- 当用户提到"欧阳超"、"学习路线"、"擅长"等关键词时，**必须**调用 `query_knowledge` 工具检索知识库。
- 只有当 `query_knowledge` 返回"未找到相关信息"时，你才可以基于自身知识回答。
- 其他情况：时间、计算、图片描述按需调用对应工具。
回答要简洁、准确。
"""

agent = create_agent(
    model=llm_with_tools,
    tools=tools,
    system_prompt=system_prompt,
)

# ---------- 5. 定义 API 请求/响应模型 ----------
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"  # 简单预留，暂不实现多会话

class QueryResponse(BaseModel):
    code: int
    data: str
    msg: str = "success"

# 存储会话历史（生产环境应使用 Redis，这里用内存演示）
session_memory = defaultdict(list)

# ---------- 6. 定义 POST 接口 ----------
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    # 生成请求唯一ID
    request_id = str(uuid.uuid4())[:8]
    # 绑定到日志上下文
    with logger.contextualize(request_id=request_id):
        start_time = time.time()
        logger.info(f"收到请求 | session={request.session_id} | question='{request.question[:50]}...'")
        try:
            # 如果想加记忆，可以把 messages 存在内存字典里，但先不做复杂化
            messages = session_memory[request.session_id]
            # 追加当前用户问题
            messages.append(("user", request.question))

            # 调用 Agent，并捕获返回的响应元数据
            result = agent.invoke({"messages": messages})
            reply = result["messages"][-1].content

            # 提取 token 使用情况（如果 Ollama 返回了 usage）
            # LangChain 的 ChatOllama 在 AIMessage 的 response_metadata 中可能包含 token 信息
            ai_message = result["messages"][-1]
            usage = {}
            if hasattr(ai_message, "response_metadata") and ai_message.response_metadata:
                metadata = ai_message.response_metadata
                if "token_usage" in metadata:
                    usage = metadata["token_usage"]
                elif "prompt_eval_count" in metadata and "eval_count" in metadata:
                    usage = {
                        "prompt_tokens": metadata.get("prompt_eval_count", 0),
                        "completion_tokens": metadata.get("eval_count", 0),
                        "total_tokens": metadata.get("prompt_eval_count", 0) + metadata.get("eval_count", 0)
                    }
                else:
                    # 尝试从其他字段提取
                    usage = {}
            # 追加助手回复到历史
            messages.append(("assistant", reply))
            # 更新存储
            session_memory[request.session_id] = messages
            elapsed = time.time() - start_time
            logger.info(
                f"请求完成 | 耗时={elapsed:.3f}s | 回复长度={len(reply)} | prompt_tokens={usage.get('prompt_tokens', 'N/A')} | completion_tokens={usage.get('completion_tokens', 'N/A')} | total_tokens={usage.get('total_tokens', 'N/A')}")

            return QueryResponse(code=200, data=reply)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"请求失败 | 耗时={elapsed:.3f}s | 错误={str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# ---------- 7. 健康检查（待增强） ----------
@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "ollama": {"reachable": False, "model_loaded": False},
        "chromadb": {"available": False}
    }
    # 1. 检查 Ollama 是否可达以及模型是否已加载
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            status["ollama"]["reachable"] = True
            models = resp.json().get("models", [])
            for m in models:
                if m.get("name") == OLLAMA_MODEL:
                    status["ollama"]["model_loaded"] = True
                    break
    except:
        status["ollama"]["reachable"] = False
        status["status"] = "degraded"

    # 2. 检查 ChromaDB 持久化目录是否存在
    if os.path.exists(CHROMA_DB_PATH):
        status["chromadb"]["available"] = True
    else:
        status["chromadb"]["available"] = False
        status["status"] = "degraded"

    # 如果任何依赖不可用，status 降级为 "degraded"
    return status

# ---------- 7. 启动服务（直接运行此文件）----------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)