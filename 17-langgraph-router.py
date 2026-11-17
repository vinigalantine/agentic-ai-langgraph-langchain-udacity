"""
    Exercise 17 - LangGraph Router

    Challenge:  You're building a text processing application that can:

                    - Reverse a string (e.g., "hello" → "olleh")
                    - Convert a string to uppercase (e.g., "hello" → "HELLO")
                    
                Your application should:

                    - Accept user input and an action type.
                    - Route to the appropriate node (reverse or upper) based on the action.
                    - Handle invalid actions gracefully.
                    
                This will be achieved by routing the input through LangGraph nodes using a conditional edge.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class State(TypedDict):
    input: str
    action: str
    output: str
    
def node_a(state: State):
    output = state["input"][::-1]

    logger.info(f"Node A - input: {state["input"]} - output: {output}")
    
    return { "output": output }

def node_b(state: State):
    output = state["input"].upper()

    logger.info(f"Node B - input: {state["input"]} - output: {output}")
    
    return { "output": output }

def node_invalid(state: State):
    wrong_action_message = f"Invalid action: {state["action"]}"
    logger.info(wrong_action_message)

    return { "output": wrong_action_message }

def routing_function(state: State):
    if state["action"].lower() == "reverse":
        return "node_a" 
    
    if state["action"].lower() == "uppercase":
        return "node_b" 
    
    return "node_invalid"

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_node(node_invalid)


workflow.add_conditional_edges(
    source=START,
    path=routing_function,
    path_map=["node_a","node_b","node_invalid"]
)

workflow.add_edge("node_a", END)
workflow.add_edge("node_b", END)
workflow.add_edge("node_invalid", END)

graph = workflow.compile()

print("Graph:")
graph.get_graph().print_ascii()

print("\n"+"="*50+"\n")

print("Reverse", graph.invoke(input={ "input": "Hello world!", "action": "reverse" }))

print("\n"+"="*50+"\n")

print("Upper", graph.invoke(input={ "input": "Hello world!", "action": "uppercase" }))

print("\n"+"="*50+"\n")

print("Invalid", graph.invoke(input={ "input": "Hello world!", "action": "lower" }))
