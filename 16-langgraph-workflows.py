"""
    Exercise 16 - LangGraph Agent Implementation

    Challenge:  None, just checking out a workflow in LangGraph
"""

from dotenv import load_dotenv
import random
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

def print_spacing():
    print("\n" + "=" * 50 + "\n")
    
# Data Processing (Squential)

class State(TypedDict):
    input: int
    output: int

def node_a(state: State) -> State:
    input_value = state["input"]
    offset = random.randint(1,10)
    output = input_value + offset
    print(f"NODE A - input: {input_value} - offset: {offset} - output: {output}")
    
    return State(output=output)

def node_b(state: State) -> State:
    input_value = state["output"]
    offset = random.randint(1,10)
    output = input_value + offset
    print(f"NODE B - input: {input_value} - offset: {offset} - output: {output}")
    
    return {"output": output}

workflow = StateGraph(state_schema=State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

graph = workflow.compile()

print("Graph:")
graph.get_graph().print_ascii()

print_spacing()

print("Graph invoke:")
print(graph.invoke(input={"input": 1}))

# Call LLMs

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

class State(TypedDict):
    question: str
    response: str
    
def model(state: State):
    question = state["question"]
    response = llm.invoke([
        SystemMessage("You are a music specialist."),
        HumanMessage(question)
    ])
    
    return {"response": response.content}

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
print(graph.invoke(input={"question": "What's the most recent music released by Skrillex?"}))
