"""
    Exercise 33 - LangChain + LangGraph HITL (Human-In-The-Loop)

    Challenge:  None, just checking out HITL (Human-In-The-Loop).
"""

import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class State(MessagesState):
    question: str
    answer: str


def print_spacing():
    print("\n" + "=" * 100 + "\n")


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


llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools([web_search])


def entry_point(state: State):
    question = state["question"]

    messages = [
        SystemMessage("You conduct web search to respond to user's questions."),
        HumanMessage(question),
    ]
    return {"messages": messages}


def agent(state: State):
    messages = state["messages"]
    agent_message = llm_with_tools.invoke(messages)
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
workflow.add_conditional_edges(source="agent", path=router, path_map=["tools", END])
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(interrupt_before=["tools"], checkpointer=memory)

print("Graph: ")
graph.get_graph().print_ascii()

input_question = {"question": "What's the capital of Brazil?"}
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(input=input_question, config=config, stream_mode="values"):
    if not event["messages"]:
        continue
    event["messages"][-1].pretty_print()

state = graph.get_state(config=config)

print_spacing()
print("State: ", state)
print("State next: ", state.next)

for event in graph.stream(input=None, config=config, stream_mode="values"):
    if not event["messages"]:
        continue
    event["messages"][-1].pretty_print()


def human_in_the_loop_run(graph: CompiledStateGraph, question: str, thread_id: str):
    input_question = {"question": question}
    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(
        input=input_question, config=config, stream_mode="values"
    ):
        if not event["messages"]:
            continue
        event["messages"][-1].pretty_print()

    human_input = input("Do you approve the tool calling? (Y/N): ")

    if human_input.lower() in ("y", "yes"):
        for event in graph.stream(input=None, config=config, stream_mode="values"):
            if not event["messages"]:
                continue
            event["messages"][-1].pretty_print()
    else:
        SystemMessage("Workflow aborted by a human").pretty_print()


print_spacing()
human_in_the_loop_run(
    graph=graph,
    question='What\'s the latest album from "twenty one pilots"?',
    thread_id=2,
)

# Human Revision

memory = MemorySaver()
graph = workflow.compile(interrupt_before=["agent"], checkpointer=memory)

print_spacing()
print("Graph: ")
graph.get_graph().print_ascii()


def human_in_the_loop_run_remastered(
    graph: CompiledStateGraph, question: str, thread_id: str
):
    input_question = {"question": question}
    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(
        input=input_question, config=config, stream_mode="values"
    ):
        if not event["messages"]:
            continue
        event["messages"][-1].pretty_print()

    human_input = input("Do you want to proceed with this question? (Y/N): ")

    if human_input.lower() in ("n", "no"):
        human_input = input("So what's the questions?")
        system_message = SystemMessage("Workflow edited by a human")
        system_message.pretty_print()
        original = graph.get_state(config).values["messages"][-1]
        edited = HumanMessage(human_input, id=original.id)
        
        graph.update_state(
            config=config,
            values={"messages": [system_message, edited]},
        )

    printed = len(graph.get_state(config).values["messages"])
    while graph.get_state(config).next:
        for event in graph.stream(input=None, config=config, stream_mode="values"):
            for message in event["messages"][printed:]:
                message.pretty_print()
            printed = len(event["messages"])


print_spacing()
human_in_the_loop_run_remastered(
    graph=graph,
    question='What\'s the latest album from "twenty one pilots"?',
    thread_id=2,
)
