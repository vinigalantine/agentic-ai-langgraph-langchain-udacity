"""
    Exercise 32 - LangChain + LangGraph Agentic RAG

    Challenge:  None, just checking out agentic RAG.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "="*100 + "\n")

embeddings_fn = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(collection_name="udacity", embedding_function=embeddings_fn)

reader = PdfReader("data/compact-guide-to-large-language-models.pdf")
pages = [
    Document(page_content=page.extract_text(), metadata={"source": reader.stream.name if hasattr(reader.stream, "name") else "pdf", "page": i})
    for i, page in enumerate(reader.pages)
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

vector_store.add_documents(documents=all_splits)

llm = ChatOpenAI(model="gpt-4o-mini")


class State(MessagesState):
    question: str
    documents: list[Document]
    search_required: str = "NO"
    answer: str


def retrieve(state: State):
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question)
    logger.info(f"Retrieve node -> Num docs retrieved: {len(retrieved_docs)}")
    return {"documents": retrieved_docs}


def evaluator(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    template = ChatPromptTemplate(
        [
            ("system", "You are an assistant for evaluating context."),
            (
                "human",
                "Use the following pieces of retrieved context to determine if an "
                "additional search is required. If the context is relevant to the "
                "question, your response should be 'NO', i.e. no need for search. "
                "If the context is not relevant, say 'YES', i.e. search the web for an answer. "
                "Don't explain anything, if it's to search the web, say YES, otherwise, NO. "
                "\n# Question: \n-> {question} "
                "\n# Context: \n-> {context} "
                "\n# Answer: ",
            ),
        ]
    )

    chain = template | llm | StrOutputParser()

    search_required = chain.invoke({"context": docs_content, "question": question})
    logger.info(f'Evaluator node -> Search required: "{search_required}"')
    
    return {"search_required": search_required }


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

    logger.info(f"Web search -> Query: {query} - Response: {response}")

    return response


llm_with_tools = llm.bind_tools([web_search])


def researcher(state: State):
    question = state["question"]
    messages = state["messages"]

    if not messages:
        messages = [
            HumanMessage(
                f"Conduct a web research to findout the answer to the following question: ```{question}```"
            )
        ]

    agent_message = llm_with_tools.invoke(messages)
    messages.append(agent_message)
    logger.info("Researcher node")
    return {"messages": messages, "answer": agent_message.content}

def augment(state: State):
    question = state["question"]
    documents = state["documents"]
    
    docs_content = "\n\n".join(doc.page_content for doc in documents)
    
    template = ChatPromptTemplate([
            ("system", "You are an assistant for question-answering tasks."),
            ("human", "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. " 
                    "Use three sentences maximum and keep the answer concise. "
                    "\n# Question: \n-> {question} "
                    "\n# Context: \n-> {context} "
                    "\n# Answer: "),
        ])
    
    messages = template.invoke({"context": docs_content, "question": question}).to_messages()

    logger.info("Augment node")
    return {"messages": messages}

def generate(state: State):
    agent_message = llm.invoke(state["messages"])
    logger.info("Generate node")
    return {"answer": agent_message.content, "messages": agent_message}

def rag_router(state: State):
    logger.info("Rag router")
    if state["search_required"].lower() == "yes":
        return "researcher"
    
    return "augment"

def tool_router(state: State):
    logger.info("Tool router")
    if state["messages"][-1].tool_calls:
        return "tools"
    
    return END

workflow = StateGraph(State)

workflow.add_node(retrieve)
workflow.add_node(augment)
workflow.add_node(generate)
workflow.add_node(evaluator)
workflow.add_node(researcher)
workflow.add_node("tools", ToolNode([web_search]))

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluator")

workflow.add_conditional_edges(
    source="evaluator",
    path=rag_router,
    path_map=["augment", "researcher"]
)

workflow.add_conditional_edges(
    source="researcher",
    path=tool_router,
    path_map=["tools", END]
)

workflow.add_edge("tools", "researcher")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

print_spacing()
print("Graph: ")
graph.get_graph().print_ascii()

output = graph.invoke(input={"question": "What is Pokemon?"})

print_spacing()
print("Answer: ")
print(output["answer"])

print_spacing()

print("Chat hist: ")
for message in output["messages"]:
    message.pretty_print()
    
print_spacing()
print("Search required: ", output["search_required"])

output = graph.invoke(input={"question": "What is Open Source model?"})

print_spacing()
print("Answer: ")
print(output["answer"])

print_spacing()

print("Chat hist: ")
for message in output["messages"]:
    message.pretty_print()
    
print_spacing()
print("Search required: ", output["search_required"])