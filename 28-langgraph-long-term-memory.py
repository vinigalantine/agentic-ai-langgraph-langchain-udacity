"""
    Exercise 28 - LangGraph long-term Memory

    Challenge:  None, just checking out long-term memory.
"""

import sqlite3
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


def print_spacing():
    print("\n" + "=" * 100 + "\n")


def run_graph(query: str, graph: StateGraph, thread_id: str):
    output = graph.invoke(
        config={"configurable": {"thread_id": thread_id}},
        input={"messages": [HumanMessage(query)]},
    )
    return output


# In-memory


def chatbot(state: MessagesState):
    agent_message = llm.invoke(state["messages"])
    return {"messages": agent_message}


workflow = StateGraph(MessagesState)

workflow.add_node(chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

checkpointer = MemorySaver()

in_memory_graph = workflow.compile(checkpointer=checkpointer)

print("In memory graph: ")
in_memory_graph.get_graph().print_ascii()

run_graph(query="Hi", graph=in_memory_graph, thread_id="1")

print("In memory state list: ")
snapshot = in_memory_graph.get_state(config={"configurable": {"thread_id": "1"}})
print(snapshot.values["messages"])

# SQLite

db_path = "data/memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

memory = SqliteSaver(conn)
external_memory_graph = workflow.compile(checkpointer=memory)

print("SQLite graph:")
external_memory_graph.get_graph().print_ascii()

run_graph(query="What's a memory?", graph=external_memory_graph, thread_id="2")

print_spacing()
print("SQLite state list:")
snapshot = external_memory_graph.get_state(config={"configurable": {"thread_id": "2"}})
print(snapshot.values["messages"])

# Querying memory

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table_name[0] for table_name in cursor.fetchall()]

print_spacing()
print("SQLite tables: ", tables)

columns_map = []
results = []

for table in tables:
    cursor.execute(f"SELECT * FROM {table}")
    results.append(cursor.fetchall())
    columns_map.append({table: [desc[0] for desc in cursor.description]})

print_spacing()
print("Columns map:", columns_map)

cursor.execute("SELECT metadata FROM checkpoints")
metadata = cursor.fetchall()

print_spacing()
print("Metadata: ", metadata)

steps = [json.loads(mtdt[0]) for mtdt in metadata]

print_spacing()
print("Steps: ", steps)
