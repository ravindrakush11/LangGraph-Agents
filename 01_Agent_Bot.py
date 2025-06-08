from typing import TypedDict, List
from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# llm = ChatOllama(model="qwen3:8b")
# Uncomment the above line to use Ollama instead of Groq

llm = ChatGroq(model = "llama3-8b-8192")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
    