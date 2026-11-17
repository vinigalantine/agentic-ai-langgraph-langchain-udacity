"""
    Exercise 18 - LangGraph State Management

    Challenge:  None, just checking out state management
"""

from random import choice
from typing import Annotated, Optional, Literal
from operator import add
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

class State(BaseModel):
    input_color: Literal["green", "yellow", "red"]
    output_color: Optional[Literal["green", "yellow", "red"]] = None
    
def color_node(state: State):
    logger.info(f"Color Node: {state.input_color}")
    return { "output_color": choice(["green", "yellow", "red"]) }

workflow = StateGraph(State)

workflow.add_node(color_node)

workflow.add_edge(START, "color_node")
workflow.add_edge("color_node", END)

graph = workflow.compile()

State(input_color="green")

output = graph.invoke({"input_color": "green"})
print("Output: ",output["output_color"])

print_spacing()

# State Machine

class State(BaseModel):
    color: Optional[Literal["green", "yellow", "red"]] = None
    repeat_mode: bool
    messages: Annotated[list[str], add] = []
    counter: Annotated[int, add] = 0
    
def green_light(state: State):
    color = "green"
    wait_time = 60
    message = f"{color} light, traffic allowed for {wait_time} seconds."
    messages = [message]
    
    return {"color": color, "messages": messages, "counter": 1}
    
def yellow_light(state: State):
    color = "yellow"
    wait_time = 5
    message = f"{color} light, traffic will stop {wait_time} seconds."
    messages = [message]
    
    return {"color": color, "messages": messages}
    
def red_light(state: State):
    color = "red"
    wait_time = 15
    message = f"{color} light, traffic stopped for {wait_time} seconds."
    messages = [message]
    messages.append("-"*20)
    return {"color": color, "messages": messages}

def should_repeat(state: State):
    repeat_mode = state.repeat_mode
    counter = state.counter
    if repeat_mode and counter <= 3:
        return "green_light"
    return END

workflow = StateGraph(State)

workflow.add_node(green_light)
workflow.add_node(yellow_light)
workflow.add_node(red_light)

workflow.add_edge(START, "green_light")
workflow.add_edge("green_light", "yellow_light")
workflow.add_edge("yellow_light", "red_light")

workflow.add_conditional_edges(
    source="red_light",
    path=should_repeat,
    path_map=["green_light", END]
)

graph = workflow.compile()

print("Graph: ")
graph.get_graph().print_ascii()

print_spacing()

print("Traffic light: ")
controller = graph.invoke({"repeat_mode": False})
for message in controller["messages"]:
    print("Message: ", message)