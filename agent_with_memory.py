###短期记忆程序###
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from datetime import datetime

# 定义工具
@tool
def get_current_time(format: str="%Y-%m-%d %H:%M:%S"):
    """获取当前日期和时间"""
    return datetime.now().strftime(format)

@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数字的和"""
    return a + b

# 初始化模型
llm = ChatOllama(
    model = "qwen3-vl:8b-instruct-q4_K_M",
    temperature=0,
)

tools = [get_current_time, add_numbers]
llm_with_tools = llm.bind_tools(tools)

#创建 Agent
agent = create_agent(
    model = llm_with_tools,
    tools=tools,
    system_prompt="你是一个有帮助的助手，可以调用工具。回答要简洁。"
)

#维护对话历史
messages = []

print("多轮对话 Agent (输入 exit 退出)")
while True:
    user_input= input("\n你：")
    if user_input == "exit":
        break
    # 将用户消息加入历史
    messages.append(("user", user_input))
    # 调用Agent
    response= agent.invoke({"messages": messages})
    # 提取助手回复
    assistant_msg = response["messages"][-1].content
    print(f"Agent: {assistant_msg}")
    # 将助手回复加入历史
    messages.append(("assistant", assistant_msg))