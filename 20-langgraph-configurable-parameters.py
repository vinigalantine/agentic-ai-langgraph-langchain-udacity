"""
    Exercise 20 - LangGraph Configurable Parameters

    Challenge:  None, just checking out configurable parameters
"""

import random
from dotenv import load_dotenv
import logging
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

# Data Processing

class State(TypedDict):
    input: int
    partial: Annotated[list[int], operator.add]
    results: Annotated[list[int], operator.add]
    
def node_a(state: State, config: RunnableConfig):
    offset = 0
    if config:
        offset = config.get("configurable", {}).get("offset",0)
    
    partial = random.randint(1,10) + offset
    output = state["input"] + partial
    
    logger.info(f"NODE A - input: {state["input"]}" 
          f" - offset: {offset}"
          f" - random: {partial-offset}"
          f" - partial: {partial}"
          f" - output: {output}")
    
    return { "partial": [partial], "results": [output]}

def node_b(state: State, config: RunnableConfig):
    offset = 0
    if config:
        offset = config.get("configurable", {}).get("offset",0)
    
    partial = random.randint(1,10) + offset
    output = state["results"][-1] + partial
    
    logger.info(f"NODE B - input: {state["input"]}" 
          f" - offset: {offset}"
          f" - random: {partial-offset}"
          f" - partial: {partial}"
          f" - output: {output}")
    
    return { "partial": [partial], "results": [output]}

workflow = StateGraph(State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

graph = workflow.compile()

config = { "configurable": { "offset": 20 }}

print("Config example: ")
print(graph.invoke(
    input= {
        "input":1
    },
    config= config
))

print_spacing()
# Call LLMs

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

class State(MessagesState):
    question: str
    
def model(state: State, config: RunnableConfig):
    question = state["question"]
    messages = state["messages"]
    name = None
    
    if config:
        name = config.get("configurable", {}).get("name")
        
    if name and not messages:
        messages.append(
            SystemMessage(f"The user's name is {name}. Before anything greet the user with the word \"SUUUUPPEEERRR!\".")
        )
    
    messages.append(HumanMessage(question))
    response = llm.invoke(messages)
    messages.append(response)
    
    return { "messages": messages }

workflow = StateGraph(State)

workflow.add_node("model", model)

workflow.add_edge(START, "model")
workflow.add_edge("model", END)

graph = workflow.compile()

result = graph.invoke(
    input={ "question": "Who is Luffy's father?" },
    config={ "configurable": { "name": "Imu" } }
)

for message in result["messages"]:
    message.pretty_print()