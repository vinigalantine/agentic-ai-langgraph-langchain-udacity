"""
    Exercise 36 - LangChain + LangGraph + MLflow Evaluate Agent.

    Challenge:  Your task is to create a LangGraph Workflow that includes:

                    - A RAG pipeline for information retrieval.
                    - An LLM-based judge for evaluation.
                    - RAGAS metrics for quality assessment.
                    - MLflow logging for observability.

                    The workflow should:

                    - Retrieve, augment, and generate answers.
                    - Evaluate the answers using RAGAS.
                    - Log performance metrics in MLflow.


    Obs: Had Opus rewrite the "base exercise" as it was kinda outdated and stuff.
    For this you will need to run the MLflow server (you'll find more info on the README.md):
    mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root ./mlartifacts
"""

import logging
from dotenv import load_dotenv
import mlflow
from mlflow import log_params, log_metrics
from openai import OpenAI, AsyncOpenAI
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JUDGE_LLM_MODEL = "gpt-4o"


def print_spacing():
    print("\n" + "=" * 100 + "\n")


llm = ChatOpenAI(model="gpt-4o-mini")
judge_llm = llm_factory(JUDGE_LLM_MODEL, client=OpenAI())

embeddings_fn = OpenAIEmbeddings(model="text-embedding-3-large")
judge_embeddings = embedding_factory(
    "openai", model="text-embedding-3-large", client=AsyncOpenAI()
)

faithfulness = Faithfulness(llm=judge_llm)
context_precision = ContextPrecision(llm=judge_llm)
context_recall = ContextRecall(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("udacity")

with mlflow.start_run(run_name="evaluate_agent_exercise") as run:
    log_params(
        {
            "embeddings_model": embeddings_fn.model,
            "llm_model": llm.model_name,
            "judge_llm_model": JUDGE_LLM_MODEL,
        }
    )

mlflow_run_id = run.info.run_id
mlflow_client = mlflow.tracking.MlflowClient()

vector_store = Chroma(collection_name="udacity", embedding_function=embeddings_fn)

file_path = "data/compact-guide-to-large-language-models.pdf"
reader = PdfReader(file_path)

pages = [
    Document(
        page_content=page.extract_text(), metadata={"source": file_path, "page": i}
    )
    for i, page in enumerate(reader.pages)
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

vector_store.add_documents(documents=all_splits)


class State(MessagesState):
    mlflow_run_id: str
    ground_truth: str
    evaluation: dict
    documents: list
    query: str
    answer: str


def retrieve(state: State):
    return {"documents": vector_store.similarity_search(state["query"])}


def augment(state: State):
    query = state["query"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    template = ChatPromptTemplate(
        [
            ("system", "You are an assistant for question-answering tasks."),
            (
                "human",
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise. "
                "\n# Question: \n-> {query} "
                "\n# Context: \n-> {context} "
                "\n# Answer: ",
            ),
        ]
    )

    messages = template.invoke({"context": docs_content, "query": query}).to_messages()

    return {"messages": messages}


def generate(state: State):
    agent_message = llm.invoke(state["messages"])
    return {"answer": agent_message.content, "messages": agent_message}


def evaluate_rag(state: State):
    query = state["query"]
    retrieved_docs = [doc.page_content for doc in state["documents"]]
    answer = state["answer"]
    ground_truth = state["ground_truth"]

    evaluation_results = {
        "faithfulness": faithfulness.score(
            user_input=query, retrieved_contexts=retrieved_docs, response=answer
        ).value,
        "context_precision": context_precision.score(
            user_input=query, retrieved_contexts=retrieved_docs, reference=ground_truth
        ).value,
        "context_recall": context_recall.score(
            user_input=query, retrieved_contexts=retrieved_docs, reference=ground_truth
        ).value,
        "answer_relevancy": answer_relevancy.score(
            user_input=query, response=answer
        ).value,
    }
    logger.info(f"Evaluation: {evaluation_results}")

    with mlflow.start_run(state["mlflow_run_id"]):
        log_metrics(evaluation_results)

    return {"evaluation": evaluation_results}


workflow = StateGraph(State)

workflow.add_node(retrieve)
workflow.add_node(augment)
workflow.add_node(generate)
workflow.add_node(evaluate_rag)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "augment")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", "evaluate_rag")
workflow.add_edge("evaluate_rag", END)

graph = workflow.compile()

print("Graph: ")
graph.get_graph().print_ascii()

print_spacing()

reference = [
    {
        "query": "What are Open source models?",
        "ground_truth": "Open-source models are AI or machine learning "
        "models whose code, architecture, and in some cases, "
        "training data and weights, are publicly available for "
        "use, modification, and distribution. They enable "
        "collaboration, transparency, and innovation by allowing "
        "developers to fine-tune, deploy, or improve them without "
        "proprietary restrictions.",
    }
]

output = graph.invoke(
    {
        "query": reference[0]["query"],
        "ground_truth": reference[0]["ground_truth"],
        "mlflow_run_id": mlflow_run_id,
    }
)

print_spacing()
for message in output["messages"]:
    message.pretty_print()

print_spacing()
print("MLflow run: ", mlflow_client.get_run(mlflow_run_id).data.metrics)
