"""
    Exercise 25 - LangGraph checkpoints

    Challenge:  None, just checking out checkpoints
    
    Obs: Already kinda touched this concept on 23
"""

from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

class State(TypedDict):
    value: Annotated[list[str], operator.add]
    
def node_a(state: State):
    return {"value": ["a"]}

def node_b(state: State):
    return {"value": ["b"]}

workflow = StateGraph(State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

graph = workflow.compile()