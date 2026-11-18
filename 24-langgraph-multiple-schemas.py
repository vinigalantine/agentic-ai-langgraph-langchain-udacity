"""
    Exercise 24 - LangGraph Multiple Schemas

    Challenge:  None, just checking out multiple schemas

    Obs: Same as 23. Going to do a different approach because I felt the exercise didn't give
    a good "grasp" on "why should we do that". Might just be in my head tho.
    Got this from Opus 5.5 btw.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TicketInput(TypedDict):
    message: str


class TicketOutput(TypedDict):
    reply: str


class TicketState(TypedDict):
    message: str
    category: str
    priority: str
    internal_note: str
    reply: str


def classify(state: TicketInput):
    text = state["message"].lower()

    if "refund" in text or "money back" in text:
        return {
            "category": "billing",
            "priority": "high",
            "internal_note": "Angry customer, refund request. Check order history for abuse before approving.",
        }

    return {
        "category": "general",
        "priority": "low",
        "internal_note": "Standard inquiry, no action needed.",
    }


def respond(state: TicketState) -> TicketOutput:
    if state["priority"] == "high":
        reply = "We're sorry for the trouble, a specialist will contact you within 24 hours."
    else:
        reply = "Thanks for reaching out! We'll get back to you soon."

    return {"reply": reply}


def build(public: bool):
    schemas = (
        {"input_schema": TicketInput, "output_schema": TicketOutput} if public else {}
    )

    workflow = StateGraph(TicketState, **schemas)

    workflow.add_node(classify)
    workflow.add_node(respond)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


ticket = {"message": "The product doesn't work. I want my money back!"}

print("Without schemas (leaks internals):")
print(build(public=False).invoke(ticket))

print("\n" + "=" * 50 + "\n")

print("With schemas (safe):")
print(build(public=True).invoke(ticket))
