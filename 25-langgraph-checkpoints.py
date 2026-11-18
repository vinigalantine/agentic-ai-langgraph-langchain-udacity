"""
    Exercise 25 - LangGraph checkpoints

    Challenge:  None, just checking out checkpoints
    
    Obs: Already kinda touched this concept on 23
"""

from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

def print_spacing():
    print("\n" + "="*100 + "\n")

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

graph.get_graph().print_ascii()

print("Graph invoke: ", graph.invoke({"value":[]}))

# Checkpoints

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = { "configurable": { "thread_id": "1" } }

print("First graph invoke after config: ", graph.invoke({ "value": [] }, config=config))

print("Second graph invoke after config: ", graph.invoke({ "value": [] }, config=config))

print_spacing()

second_config = { "configurable": { "thread_id": "2" } }

print("New thread id: ", graph.invoke({ "value": [] }, config=second_config))

print_spacing()

print(
    "State history: \n",
    *(f"{item} \n\n" for item in graph.get_state_history(config))
)