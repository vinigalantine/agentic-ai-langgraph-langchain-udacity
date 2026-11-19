"""
    Exercise 27 - Langchain + LangGraph Calling APIs

    Challenge:  None, just checking out calling APIs.
"""
# import os
import requests
import random
import logging
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "="*100 + "\n")

@tool
def get_random_ricknmorty_character_tool() -> dict:
    """Get a random Rick&Morty character.

    Returns:
        dict: Information about a random Rick&Morty character.
    """

    character_id = random.randint(1, 826)
    response = requests.get(f"https://rickandmortyapi.com/api/character/{character_id}")

    logger.info(
        f"Random Rick&Morty character -> Character id: {character_id} - Response: \n{response.json()}"
    )

    return response.json()


@tool
def get_ricknmorty_episode_tool(episode_id: int) -> dict:
    """Get information on a Rick&Morty episode.

    Args:
        episode_id (int): The id of the episode

    Returns:
        dict: Information about a Rick&Morty episode.
    """
    response = requests.get(f"https://rickandmortyapi.com/api/episode/{episode_id}")

    logger.info(
        f"Rick&Morty episode info -> Episode id: {episode_id} - Response: \n{response.json()}"
    )

    return response.json()

print_spacing()
print("Random Rick&Morty character: ", get_random_ricknmorty_character_tool.invoke({}))

# We can use it without an API key but it has a limit, if you have a key comment the line below and uncomment the other one
# if you are going to use the API don't forget to uncomment the "import os" on the top of the file
tavily_client = TavilyClient()
# tavily_client = TavilyClient(
#     api_key=os.getenv("TAVILY_API_KEY")
# )


@tool
def web_search(query: str) -> dict:
    """Do a websearch on Tavily

    Args:
        query (str): Query to be executed

    Returns:
        dict: Result of the web search.
    """
    response = tavily_client.search(query)

    logger.info(f"Web search -> Query: {query} - Response: {response}")

    return response


print_spacing()
print(web_search.invoke({"query": "Who is the lead singer of Duran Duran?"}))

tools = [get_random_ricknmorty_character_tool, get_ricknmorty_episode_tool, web_search]

llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)


def agent(state: MessagesState):
    agent_message = llm_with_tools.invoke(state["messages"])
    return {"messages": agent_message}


def router(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"

    return END


workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(source="agent", path=router, path_map=["tools", END])

workflow.add_edge("tools", "agent")

graph = workflow.compile()

print("Graph: ")
graph.get_graph().print_ascii()

messages = [
    SystemMessage(
        "You are a web researcher focused on Rick and Morty.\n"
        "When the user asks for a random character, follow these steps in order:\n"
        "1. Get a random character with the character tool.\n"
        "2. Take the FIRST URL in the character's `episode` list; the number at the "
        "end of that URL is the episode id. Get that episode with the episode tool.\n"
        "3. Search the web for a summary of that episode, using its name and code "
        "(e.g. 'Rick and Morty S01E01 Pilot summary').\n"
        "4. Answer using only the information returned by the tools. If a search "
        "returns nothing useful, say so instead of guessing.\n\n"
        "Format your final answer as:\n"
        "- Character: name, status, species, origin, current location\n"
        "- First appearance: episode name, code, air date\n"
        "- Episode summary: 3-5 sentences\n"
        "- Sources: the URLs you used"
    ),
    HumanMessage("Give me a random character from Rick&Morty."),
]

result = graph.invoke(input={"messages": messages})

print_spacing()

for message in result["messages"]:
    message.pretty_print()
