"""
Short-term memory agent with tool calling and multi-turn conversation.

This agent uses LangChain to bind tools (time query, addition) and maintains
conversation history within a session.
"""

from datetime import datetime

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


# ---------- Tools ----------
@tool
def get_current_time(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current date and time in the given format."""
    return datetime.now().strftime(format_str)


@tool
def add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


# ---------- Agent Setup ----------
def create_agent_with_tools():
    """Initialize the LangChain agent with time and math tools."""
    llm = ChatOllama(
        model="qwen3-vl:8b-instruct-q4_K_M",
        temperature=0,
    )

    tools = [get_current_time, add_numbers]
    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant that can call tools. "
            "Keep responses concise and accurate."
        ),
    )
    return agent


# ---------- Main Loop ----------
def run_conversation():
    """Start an interactive multi-turn conversation session."""
    agent = create_agent_with_tools()
    conversation_history = []

    print("Multi-turn Agent (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        conversation_history.append(("user", user_input))
        response = agent.invoke({"messages": conversation_history})
        assistant_msg = response["messages"][-1].content
        print(f"Agent: {assistant_msg}")
        conversation_history.append(("assistant", assistant_msg))


if __name__ == "__main__":
    run_conversation()