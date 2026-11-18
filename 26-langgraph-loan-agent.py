"""
Exercise 26 - LangGraph Loan Agent

Challenge:  You’re developing an AI-powered Loan agent.

            The workflow should:

                - Interpret user queries intelligently.
                - Dynamically calls tools for loan calculation
                - Track negotiation status and terminates gracefully.
                - Route the workflow using intelligent conditional edges.

            Your agent will calculate the maximum amount a customer can request based on their profile. If it exceeds, it will mark the status as lost and will explain the reason to the customer. If it meets, make as won, and notify the customer a human representative will contact them.
            By the end of this exercise, you will have learned:

                - Config
                - Limiting Messages (RemoveMessages and summary)
                - Command (Updating State from a Tool)
                - Checkpoints (with thread_id)
"""

import re
import uuid
import logging
from dotenv import load_dotenv
from typing import Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_spacing():
    print("\n" + "=" * 100 + "\n")


class State(MessagesState):
    query: str
    negotiation_status: Literal["in_progress", "won", "lost"]


def summarize_conversation(messages) -> AIMessage:
    system_message = SystemMessage(
        content="Please go over the messages and summarize them in a short, simple and compact form."
        ' And start your message with "Summarization:"',
        id=str(uuid.uuid4()),
    )

    return llm.invoke(messages + [system_message])


@tool
def calculate_max_loan(
    age: Annotated[int, "greater than 17"],
    income: Annotated[int, "greater than 0"],
    credit_score: Annotated[int, "between 0 and 1000"],
) -> int:
    """Calculate the max loan for a customer, considering age, income and credit score.

    Args:
        age (int): Age of the customer
        income (int): Income of the customer
        credit_score (int): Credit score of the customer

    Returns:
        int: Max loan that the customer can get.
    """
    loan = income * 10

    if not 25 <= age <= 60:
        loan = loan * 0.5

    if credit_score >= 700:
        loan = loan * 1.2

    logger.info(
        f"Calculate loan tool ->"
        f" age: {age} - income: {income} - credit_score: {credit_score}"
        f" - loan: {loan}"
    )
    return round(loan)


@tool
def update_negotiation_status(
    status: Literal["won", "lost"], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Updates the negotiation status.

    Args:
        status (Literal["won", "lost"]): Status of the loan
        tool_call_id (str): Tool call id

    Returns:
        Command: Updates "negotiation_status" in the graph state.
    """
    logger.info(
        f"Update negotiation tool -> status: {status} - tool_call_id: {tool_call_id}"
    )

    return Command(
        update={
            "negotiation_status": status,
            "messages": [
                ToolMessage(
                    content='The "negotiation_status" updated successfully.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def entry_point(state: State, config: RunnableConfig):
    history = state["messages"]
    messages = []
    customer_name = config.get("configurable", {}).get("customer_name", "customer")
    age = config.get("configurable", {}).get("age", 18)
    income = config.get("configurable", {}).get("income", 1000)
    credit_score = config.get("configurable", {}).get("credit_score", 500)

    if not history:
        prompt = (
            "You are a Loan Agent analysing a customer loan request.\n"
            "- Understand how much the customer needs.\n"
            "- Calculate the maximum loan amount using one of the provided tools.\n"
            "- If the requested amount exceeds the limit, update the negotiation to lost.\n"
            "- If the requested amount meets the limit, update the negotiation to won.\n"
            "- Update status as won or lost based on the negotiation.\n"
            "Terminate gracefully:\n"
            "  - If won: Notify that a human representative will contact them.\n"
            "  - If lost: Thank the customer for their time and explain why the loan was refused.\n\n"
            "Details about the customer you're negotiating with:\n"
            f"\n- customer_name: {customer_name}"
            f"\n- customer_age: {age}"
            f"\n- customer_income: {income}"
            f"\n- customer_credit_score: {credit_score}"
        )

        messages.append(SystemMessage(content=prompt, name="system"))

    logger.info(f"Number of messages: {len(history)}")

    if len(history) >= 7:
        messages.extend(RemoveMessage(id=message.id) for message in history if message.name != "system")
        messages.append(summarize_conversation(history[1:]))

    logger.info(f"Customer query: {state['query']}")

    human_message = HumanMessage(
        content=state["query"],
        name=re.sub(r"\s", "_", customer_name.strip()).lower(),
        id=str(uuid.uuid4()),
    )
    messages.append(human_message)

    return {"messages": messages, "negotiation_status": "in_progress"}


def loan_agent_node(state: State):
    agent_message = loan_agent.invoke(state["messages"])
    # this is a pt-br joke "agiota" = "loan shark"
    agent_message.name = "agiota"

    return {"messages": agent_message}


def negotiation_router(state: State):
    logger.info(f"Router - negotiation_status: {state['negotiation_status']}")

    if state["negotiation_status"] != "in_progress":
        return END

    if state["messages"][-1].tool_calls:
        return "tools"

    return END


tools = [calculate_max_loan, update_negotiation_status]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

loan_agent = llm.bind_tools(tools=tools)

workflow = StateGraph(State)

workflow.add_node("entry_point", entry_point)
workflow.add_node("loan_agent", loan_agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "loan_agent")
workflow.add_conditional_edges(
    source="loan_agent", path=negotiation_router, path_map=["tools", END]
)
workflow.add_edge("tools", "loan_agent")

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

print("Graph: ")
graph.get_graph().print_ascii()

print_spacing()

query = "I want a $1.000.000 loan to build a ship."

inputs = {"query": query}

thread_id = str(uuid.uuid4())
config = {
    "thread_id": thread_id,
    "customer_name": "Franky",
    "age": 30,
    "income": 100,
    "credit_score": 700,
}

result = graph.invoke(input=inputs, config=config)

for message in result["messages"]:
    message.pretty_print()

print("Negotiation status: ", result["negotiation_status"])

print_spacing()

states = list(
    graph.get_state_history(config={"configurable": {"thread_id": thread_id}})
)

print("State history: ")
print(*(f"\nState: {state}\n" for state in states))

print_spacing()

print(
    "First state in the list, last message: ", states[0].values["messages"][-1].content
)

for message in states[0].values["messages"]:
    message.pretty_print()
