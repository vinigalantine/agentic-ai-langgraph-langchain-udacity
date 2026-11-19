"""
    Exercise 29 - LangGraph Langmem

    Challenge:  None, just checking out LangMem.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.messages import HumanMessage

load_dotenv()


def print_spacing():
    print("\n" + "=" * 100 + "\n")


store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

tools = [
    create_manage_memory_tool(namespace=("memories",)),
    create_search_memory_tool(namespace=("memories",)),
]

agent = create_agent(model="openai:gpt-4o-mini", tools=tools, store=store)

print("Agent graph: ")
agent.get_graph().print_ascii()

print_spacing()

print("Agent input schema: ", agent.get_input_jsonschema())

output = agent.invoke(
    input={"messages": [HumanMessage("What's my favorite color?")]},
    config={"configurable": {"thread_id": "1"}},
)

print_spacing()

for message in output["messages"]:
    message.pretty_print()

print_spacing()

messages = output["messages"]

messages.append(HumanMessage("Ok. Remember that I like red."))

output = agent.invoke(
    input={"messages": messages}, config={"configurable": {"thread_id": "1"}}
)

print_spacing()

for message in output["messages"]:
    message.pretty_print()

print_spacing()

output = agent.invoke(
    input={"messages": [HumanMessage("What's my favorite color?")]},
    config={"configurable": {"thread_id": "2"}},
)

print_spacing()

print("Final output:\n")
for message in output["messages"]:
    message.pretty_print()

print_spacing()
