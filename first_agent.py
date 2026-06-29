"""
Multimodal AI Agent (Terminal Version)
======================================

A LangChain-based agent with tools for:
- Time querying
- Math calculation
- Image description (via Qwen3-VL)
- RAG retrieval (via Chroma)

Requirements:
- Ollama service running with qwen3-vl:8b-instruct-q4_K_M
- Chroma vector DB initialized at ./chroma_db
"""

import os
import base64
from datetime import datetime

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# ---------- Configuration ----------
OLLAMA_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_PATH = "./chroma_db"
OLLAMA_BASE_URL = "http://localhost:11434"


# ---------- Tools ----------
@tool
def get_current_time() -> str:
    """
    Return current date and time in a natural Chinese expression.
    Example: "2026年6月29日 晚上8点30分15秒"
    """
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

    return (
        f"{now.year}年{now.month}月{now.day}日 "
        f"{am_pm}{hour_12}点{now.minute}分{now.second}秒"
    )


@tool
def add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


@tool
def describe_image(image_path: str) -> str:
    """
    Describe an image using Qwen3-VL.

    Args:
        image_path: Local image path (absolute or relative).

    Returns:
        Description string or error message.
    """
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return f"❌ Image not found: {abs_path}"

    try:
        with open(abs_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"❌ Failed to read image: {e}"

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Please describe this image in detail.",
                "images": [img_b64],
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
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
)


@tool
def query_knowledge(question: str) -> str:
    """Retrieve relevant information from the local knowledge base."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    print(f"[DEBUG] Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc.page_content[:100]}...")

    if not docs:
        return "No relevant information found."

    content = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Based on the following information, answer the question:\n\n"
        f"{content}\n\n"
        f"Question: {question}\n"
        f"Answer: "
    )
    response = llm.invoke(prompt)
    return response.content


# ---------- Agent Initialization ----------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)

tools = [get_current_time, add_numbers, describe_image, query_knowledge]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """
You are a helpful assistant that can call tools to answer questions.

**Rules**:
- When the user asks about "Ouyang Chao", "learning path", "expertise", or similar topics, you **must** call `query_knowledge` first.
- Only if `query_knowledge` returns "No relevant information found" may you answer from your own knowledge.
- For time, math, or image description, use the corresponding tools.
- Keep responses concise and accurate.
"""

agent = create_agent(
    model=llm_with_tools,
    tools=tools,
    system_prompt=system_prompt,
)


# ---------- Main Loop ----------
def run_conversation():
    """Start a multi-turn conversation session with memory."""
    messages = []
    print("Multi-turn Agent (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        messages.append(("user", user_input))
        response = agent.invoke({"messages": messages})
        assistant_msg = response["messages"][-1].content
        print(f"Agent: {assistant_msg}")
        messages.append(("assistant", assistant_msg))


if __name__ == "__main__":
    run_conversation()