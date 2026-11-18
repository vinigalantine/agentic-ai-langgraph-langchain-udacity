"""
    Exercise 23 - LangGraph Limiting Messages

    Challenge:  None, just checking out limiting messages
    
    Obs: Going to do a different approach because I felt the exercise didn't give 
    a good "grasp" on "why should we do that". Might just be in my head tho.
    Got this from Sonnet 5 btw. (Was kinda good because introduced the checkpointer idea earlier)
"""

import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, trim_messages

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

MAX_TOKENS = 300
KEEP_LAST_N = 6

class State(MessagesState):
    max_tokens: int
    filtered_messages: list[AnyMessage]
    
def removal_filter(state: State):
    messages_to_delete = [
        RemoveMessage(id=message.id) for message in state["messages"][:-KEEP_LAST_N]
    ]
    return { "messages": messages_to_delete}
    
def trim_filter(state: State):
    filtered_messages = trim_messages(
        messages=state["messages"],
        max_tokens=state["max_tokens"],
        strategy="last",
        token_counter=llm,
        include_system=False,
        allow_partial=False
    )
    return { "filtered_messages": filtered_messages }

def llm_node(state: State):
    return { "messages": llm.invoke(state["filtered_messages"]) }

workflow = StateGraph(State)
workflow.add_node("removal_filter", removal_filter)
workflow.add_node("trim_filter", trim_filter)
workflow.add_node("llm_node", llm_node)

workflow.add_edge(START, "removal_filter")
workflow.add_edge("removal_filter", "trim_filter")
workflow.add_edge("trim_filter", "llm_node")
workflow.add_edge("llm_node", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
config = { "configurable": { "thread_id": "straw-hat-crew" } }

turns = [
    "Write a short opening paragraph for a story about a lighthouse keeper.",
    "Now add a second paragraph introducing a mysterious ship.",
    "Describe the keeper's reaction to seeing the ship up close.",
    "Add a flashback about why the keeper took this job.",
    "Introduce a second character who arrives on the ship.",
    "Have the two characters have a tense conversation.",
    "Reveal a secret about the ship's cargo.",
    "Escalate the conflict between the characters.",
    "Write a resolution to the conflict.",
    "Write a closing paragraph that wraps up the story.",
]

full_history = []

for i, turn in enumerate(turns, start=1):
    full_history.append(HumanMessage(content=turn))
    full_response = llm.invoke(full_history)
    full_history.append(full_response)
    full_tokens = full_response.response_metadata["token_usage"]["total_tokens"]
    
    result = graph.invoke({ "messages": [HumanMessage(content=turn)], "max_tokens": MAX_TOKENS }, config=config )
    trimmed_response = result["messages"][-1]
    trimmed_tokens = trimmed_response.response_metadata["token_usage"]["total_tokens"]
    
    stored_count = len(checkpointer.get(config)["channel_values"]["messages"])
    logger.info(f"Turn {i}: full={full_tokens}, trimmed={trimmed_tokens}, stored_in_thread={stored_count}")