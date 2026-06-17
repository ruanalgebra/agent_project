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
    abs_path = os.path.abspath(image_path)
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
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
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
    try:
        # 这里简单处理：每次请求新建一个消息列表（为了演示无状态）
        # 如果想加记忆，可以把 messages 存在内存字典里，但先不做复杂化
        messages = session_memory[request.session_id]
        # 追加当前用户问题
        messages.append(("user", request.question))
        # 调用 Agent
        result = agent.invoke({"messages": messages})
        reply = result["messages"][-1].content
        # 追加助手回复到历史
        messages.append(("assistant", reply))
        # 更新存储
        session_memory[request.session_id] = messages
        return QueryResponse(code=200, data=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- 7. 启动服务（直接运行此文件）----------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)