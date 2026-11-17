"""
Exercise 19 - LangGraph Reducers

Challenge:  None, just checking out reducers
"""

import random
import logging
import operator

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_spacing():
    print("\n" + "=" * 50 + "\n")


help(operator.add)
print(
    "operator.add example (operator.add([1,2,3],[4,5,6])): ",
    operator.add([1, 2, 3], [4, 5, 6]),
)

print_spacing()
# Copied from exercise 16 -> Data Processing (Squential)
# Modifing for Data Processing Parallel with a Reducer


class State(TypedDict):
    input: int
    output: Annotated[list[int], operator.add]  # <- look it changed


def node_a(state: State) -> State:
    input_value = state["input"]
    offset = random.randint(1, 10)
    output = input_value + offset
    logger.info(f"NODE A - input: {input_value} - offset: {offset} - output: {output}")

    return {"output": [output]}  # <- look it changed too


def node_b(state: State) -> State:
    input_value = state["input"]
    offset = random.randint(1, 10)
    output = input_value + offset
    logger.info(f"NODE B - input: {input_value} - offset: {offset} - output: {output}")

    return {"output": [output]}  # <- look it changed too


workflow = StateGraph(state_schema=State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge(START, "node_b")
workflow.add_edge("node_a", END)
workflow.add_edge("node_b", END)

graph = workflow.compile()

print("Graph:")
graph.get_graph().print_ascii()

print_spacing()

print("Graph invoke:")
print(graph.invoke(input={"input": 1}))

# Call LLMs -> Now with Custom State and Langgraph operator

operator.add(
    [
        SystemMessage("You are a shipwright for the Straw Hat pirates. Luffy's crew."),
        HumanMessage("SUUUUUPEEERR!"),
    ],
    [AIMessage("SUUUPPEEEERRRRRRR!")],
)

add_messages(
    left = [
        SystemMessage("You are a shipwright for the Straw Hat pirates. Luffy's crew."),
        HumanMessage("SUUUUUPEEERR!"),
    ],
    right=AIMessage("SUUUPPEEEERRRRRRR!"),
)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def model(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return { "messages": response }

workflow = StateGraph(State)

workflow.add_node("model", model)

workflow.add_edge(START, "model")
workflow.add_edge("model", END)

graph = workflow.compile()

print_spacing()

print("Graph:")
graph.get_graph().print_ascii()

print_spacing()

print("Graph invoke:")
print(graph.invoke(input={"messages": [HumanMessage("What's the best ship in One Piece?")]}))

print_spacing()
# Call LLMs with LangGraph

def model(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return { "messages": response }

workflow = StateGraph(MessagesState)

workflow.add_node("model", model)

workflow.add_edge(START, "model")
workflow.add_edge("model", END)

graph = workflow.compile()

print("Graph:")
graph.get_graph().print_ascii()

print_spacing()

print("Graph invoke:")
result = graph.invoke(input={"messages": [HumanMessage("What's the best ship in One Piece?")]})
print(result)

for message in result["messages"]:
    message.pretty_print()