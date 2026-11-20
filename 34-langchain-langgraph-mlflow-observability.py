"""
    Exercise 34 - LangChain + LangGraph + MLflow observability.

    Challenge:  None, just checking out MLflow and observability.

    Obs: For this you will need to run the MLflow server (you'll find more info on the README.md):
    mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root ./mlartifacts
"""

import logging
import os
import mlflow
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "=" * 100 + "\n")

tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

experiment = mlflow.set_experiment("udacity")

@mlflow.trace
def add(a, b):
    return a + b

add(1, 2)

llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def web_search(query: str) -> dict:
    """Do a web search on Tavily

    Args:
        query (str): Query to be executed

    Returns:
        dict: Result of the web search.
    """
    # Can be used without API key but has a limit
    # tavily_client = TavilyClient()
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query)

    logger.info(f"Web search -> Query: {query}")

    return response

class State(MessagesState):
    question: str
    answer: str
    
llm_with_tools = llm.bind_tools([web_search])

def entry_point(state: State):
    question = state["question"]
    messages = [
        SystemMessage("You conduct web search to respond to user's questions."),
        HumanMessage(question)
    ]
    
    return {"messages": messages}

def agent(state: State):
    agent_message = llm_with_tools.invoke(state["messages"])
    return {"messages": agent_message, "answer": agent_message.content}

def router(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    
    return END

workflow = StateGraph(State)

workflow.add_node(entry_point)
workflow.add_node(agent)
workflow.add_node("tools", ToolNode([web_search]))

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "agent")
workflow.add_conditional_edges(
    source="agent",
    path=router,
    path_map=["tools", END]
)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(interrupt_before=["tools"], checkpointer=memory)

print("Graph: ")
graph.get_graph().print_ascii()

print_spacing()

mlflow.langchain.autolog()

input_question = {"question": 'What\'s the latest album from "twenty one pilots"?'}
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(input=input_question, config=config, stream_mode="values"):
    if not event["messages"]:
        continue
    
    event["messages"][-1].pretty_print()
    
state = graph.get_state(config=config)
print("State: ", state.next)

for event in graph.stream(input=None, config=config, stream_mode="values"):
    if not event["messages"]:
        continue
    
    event["messages"][-1].pretty_print()