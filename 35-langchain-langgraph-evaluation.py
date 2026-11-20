"""
    Exercise 35 - LangChain + LangGraph evaluation.

    Challenge:  None, just checking out evaluation.
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from ragas import messages as ragas_messages
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    ToolCallAccuracy,
    AgentGoalAccuracyWithReference,
)
from ragas.integrations.langgraph import convert_to_ragas_messages

load_dotenv()


def print_spacing():
    print("\n" + "=" * 100 + "\n")


documents = [
    Document(
        page_content="Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
        metadata={"company": "Meta", "topic": "llama"},
    ),
    Document(
        page_content="Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
        metadata={"company": "Nvidia", "topic": "acquisition"},
    ),
    Document(
        page_content="Google is bringing Gemini to all older Pixel Buds",
        metadata={"company": "Google", "topic": "gemini"},
    ),
    Document(
        page_content="The first Intel Battlemage GPU benchmarks have leaked",
        metadata={"company": "Intel", "topic": "gpu"},
    ),
    Document(
        page_content="Dell partners with Nvidia to accelerate AI adoption in telecoms",
        metadata={"company": "Dell", "topic": "partnership"},
    ),
]

ids = ["id1", "id2", "id3", "id4", "id5"]

vector_store = Chroma(collection_name="udacity", embedding_function=OpenAIEmbeddings())
vector_store.add_documents(documents=documents, ids=ids)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context: \n\n{context}\n\nQuestion: {query}"
)

chain = prompt | llm | StrOutputParser()


def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


query = "Who is partnering with Nvidia?"
relevant_docs = retriever.invoke(query)
print("Chain test: ", chain.invoke({"context": format_docs(relevant_docs), "query": query}))

print_spacing()

# RAGAS Evaluation

sample_queries = [
    "What is Meta's latest development in AI?",
    "Which company did Nvidia acquire?",
    "What AI feature is Google adding to older Pixel Buds?",
    "What recent information has leaked about Intel GPUs?",
    "Which company did Dell partner with to accelerate AI adoption?",
]

expected_responses = [
    "Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
    "Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
    "Google is bringing Gemini to all older Pixel Buds",
    "The first Intel Battlemage GPU benchmarks have leaked",
    "Dell partners with Nvidia to accelerate AI adoption in telecoms",
]

dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = chain.invoke({"context": format_docs(relevant_docs), "query": query})

    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [doc.page_content for doc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluator_llm = llm_factory("gpt-4o-mini", client=OpenAI())

context_recall = ContextRecall(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
factual_correctness = FactualCorrectness(llm=evaluator_llm)

print_spacing()
for row in dataset:
    print(row["user_input"])
    print(
        " context_recall: ",
        context_recall.score(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        ).value,
    )
    print(
        " faithfulness: ",
        faithfulness.score(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            response=row["response"],
        ).value,
    )
    print(
        " factual_correctness: ",
        factual_correctness.score(
            response=row["response"], reference=row["reference"]
        ).value,
    )

# LangGraph workflow

pokemon_types_map = {
    "pikachu": "electric",
    "eevee": "normal",
    "bulbasaur": "grass/poison",
    "squirtle": "water",
    "charizard": "fire/flying",
    "jigglypuff": "normal/fairy",
    "meowth": "normal",
    "psyduck": "water",
    "machamp": "fighting",
    "gengar": "ghost/poison",
    "alakazam": "psychic",
    "snorlax": "normal",
    "dragonite": "dragon/flying",
}


@tool
def get_pokemon_type(pokemon_name: str) -> str:
    """Get pokemon type of a pokemon

    Args:
        pokemon_name (str): The name of the Pokémon

    Returns:
        str: The pokémon type
    """

    try:
        pokemon_name = pokemon_name.lower().strip()

        if pokemon_name not in pokemon_types_map:
            raise KeyError(
                f'Pokémon "{pokemon_name}" not found. Available Pokémon {", ".join(pokemon_types_map.keys())}'
            )

        return pokemon_types_map[pokemon_name]
    except Exception as e:
        raise Exception(f"Error fetching Pokémon type: {e}")


llm_with_tools = llm.bind_tools([get_pokemon_type])


def agent(state: MessagesState):
    return {"messages": llm_with_tools.invoke(state["messages"])}


def router(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"

    return END


workflow = StateGraph(MessagesState)

workflow.add_node(agent)
workflow.add_node("tools", ToolNode([get_pokemon_type]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(source="agent", path=router, path_map=["tools", END])

workflow.add_edge("tools", "agent")

graph = workflow.compile()

print_spacing()
print("Graph: ")
graph.get_graph().print_ascii()

result = graph.invoke({"messages": [HumanMessage("What's Gengar's type?")]})

print_spacing()
for message in result["messages"]:
    message.pretty_print()


ragas_trace = convert_to_ragas_messages(result["messages"])

print_spacing()
print("Ragas trace: ", ragas_trace)

# Evaluate tool use

tool_score = ToolCallAccuracy().score(
    user_input=ragas_trace,
    reference_tool_calls=[
        ragas_messages.ToolCall(
            name="get_pokemon_type", args={"pokemon_name": "gengar"}
        )
    ],
)

print_spacing()
print("ToolCallAccuracy score: ", tool_score.value)

# Evaluate Agent Goal

goal_score = AgentGoalAccuracyWithReference(llm=evaluator_llm).score(
    user_input=ragas_trace, reference="Gengar's type is ghost/poison"
)

print_spacing()
print("AgentGoalAccuracyWithReference score: ", goal_score.value)
