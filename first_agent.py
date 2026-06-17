"""
多模态AI智能体 (Multimodal AI Agent)
====================================
技术栈：LangChain + Ollama + Chroma + Qwen3-VL

功能：
- 视觉理解：通过qwen3-vl描述图片内容
- 知识库问答：基于Chroma检索本地文档（RAG）
- 工具调用：时间查询、数学计算
- 多轮对话：带记忆的Agent交互

运行前提：
- Ollama服务已启动，且已拉取 qwen3-vl:8b-instruct-q4_K_M
- Chroma向量库已初始化（./chroma_db）
"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from datetime import datetime
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import requests
import base64
import os

# 1. 定义工具（Agent 的“手脚”）
@tool
def get_current_time() -> str:
    """获取当前日期和时间，返回中文自然表达"""
    now = datetime.now()
    hour_12 = now.hour % 12
    if hour_12 == 0:
        hour_12 = 12
    am_pm = "上午" if now.hour < 12 else "晚上" if now.hour < 18 else "晚上"
    # 简单判断：0-11为上午，12-23为晚上（这里可以根据你的习惯调整）
    if 5 <= now.hour < 12:
        am_pm = "早上" if now.hour < 8 else "上午"
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
    """
    分析图片内容并返回描述。
    参数 image_path: 本地图片的绝对路径（例如 C:\\Users\\xxx\\photo.jpg）
    """
    # 1. 将相对路径转换为绝对路径（基于当前工作目录）
    abs_path = os.path.abspath(image_path)

    # 2. 检查文件是否存在
    if not os.path.exists(abs_path):
        return f"❌ 图片文件不存在：{abs_path}。请检查路径是否正确。"

    # 3. 读取图片并转为 base64
    try:
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"❌ 读取图片失败：{str(e)}"

    # 4. 调用 Ollama 的多模态接口
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3-vl:8b-instruct-q4_K_M",
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

# 加载已有的向量库
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@tool
def query_knowledge(question: str) -> str:
    """从本地知识库中检索相关信息"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    #调试阶段代码
    print(f"[DEBUG] 检索到 {len(docs)} 个片段：")
    for i, doc in enumerate(docs):
        print(f"  {i + 1}. {doc.page_content[:100]}...")

    if not docs:
        return "知识库中没有找到相关信息。"
    content = "\n\n".join([doc.page_content for doc in docs])
    #用大模型生成简洁答案
    prompt = f"根据以下信息回答问题: \n\n{content}\n\n问题: {question}\n答案: "
    respon = llm.invoke(prompt)
    return respon.content

# 2. 连接本地模型
llm = ChatOllama(
    model="qwen3-vl:8b-instruct-q4_K_M",
    temperature=0,
)

# 3. 将工具绑定到模型（关键步骤：让模型知道可以调用这些函数）
tools = [get_current_time, add_numbers, describe_image, query_knowledge]
llm_with_tools = llm.bind_tools(tools)


# system_prompt = """你是一个有帮助的助手，可以调用工具来回答问题。
# **重要规则**：当用户问题中包含“根据知识库”或询问特定人物（如欧阳超）的详细信息时，你必须调用 `query_knowledge` 工具，不要直接回答。
# 其他情况：时间、计算、图片描述按需调用对应工具。
# 回答要简洁、准确。"""

system_prompt = """
你是一个有帮助的助手，可以调用工具来回答问题。

**强制规则**：
- 当用户问题涉及“欧阳超”、“他的”、“学习路线”、“擅长”等与知识库中人物/事件相关的内容时，**必须**先调用 `query_knowledge` 工具获取信息，不要直接回答。
- 只有当 `query_knowledge` 返回“未找到相关信息”时，你才可以基于自身知识回答。
- 其他情况：时间、计算、图片描述按需调用对应工具。

回答要简洁、准确。
"""

# 4. 创建 Agent (自动处理函数调用循环)
agent = create_agent(
    model = llm_with_tools,
    tools = tools,
    system_prompt = system_prompt,
)
#---------多轮对话(带记忆)---------
messages = []
print("多轮对话 Agent (输入 exit 退出)")
while True:
    user_input = input("\n你：")
    if user_input.lower() == "exit":
        break
    messages.append(("user", user_input))
    response = agent.invoke({"messages": messages})
    assistant_msg = response["messages"][-1].content
    print(f"Agent: {assistant_msg}")
    messages.append(("assistant", assistant_msg))
