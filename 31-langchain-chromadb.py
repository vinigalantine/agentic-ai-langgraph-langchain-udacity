"""
    Exercise 31 - LangChain and Chromadb

    Challenge:  None, just checking out chromadb.
"""

import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import numpy as np

load_dotenv()

def print_spacing():
    print("\n" + "="*100 + "\n")

sentence_list = [
    "Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
    "Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
    "Google is bringing Gemini to all older Pixel Buds",
    "The first Intel Battlmage GPU benchmarks have leaked",
    "Dell partners with Nvidia to accelerate AI adoption in telecoms",
]
ids = ["id1", "id2", "id3", "id4", "id5"]

chroma_client = chromadb.Client()

# To persist
chroma_client = chromadb.PersistentClient(path="data/chromadb")

collection = chroma_client.create_collection(name="udacity")

collection.add(documents=sentence_list, ids=ids)

print("Collection count: ", collection.count())

print_spacing()

print("Collection peek: ", collection.peek(2))

print_spacing()

print("Collection query: ", collection.query(query_texts=["gadget"], n_results=2, include=["metadatas", "documents", "distances"]))

print_spacing()

# Changing models

embeddings_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

embeddings = embeddings_fn(sentence_list)

print("Embedding len: ", len(embeddings))

print_spacing()

print(np.dot(embeddings[1], embeddings[4]))
print(sentence_list[1])
print(sentence_list[4])

print_spacing()

embeddings_fn = embedding_functions.OpenAIEmbeddingFunction()

print("Embedding model name: ", embeddings_fn.model_name)

chroma_client.delete_collection(name="udacity")

collection = chroma_client.create_collection(name="udacity", embedding_function=embeddings_fn)

collection.add(documents=sentence_list,ids=ids)

print_spacing()

print("Collection query: ", collection.query(query_texts=["gadget"], n_results=2, include=["metadatas", "documents", "distances"]))

print_spacing()

# Using LangChain
chroma_client.delete_collection(name="udacity")

vector_store = Chroma(collection_name="udacity", embedding_function=OpenAIEmbeddings())

documents = [
    Document(
        page_content="Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
        metadata={"company":"Meta", "topic": "llama"}
    ),
    Document(
        page_content="Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
        metadata={"company":"Nvidia", "topic": "acquisition"}
    ),
    Document(
        page_content="Google is bringing Gemini to all older Pixel Buds",
        metadata={"company":"Google", "topic": "gemini"}
    ),
    Document(
        page_content="The first Intel Battlmage GPU benchmarks have leaked",
        metadata={"company":"Intel", "topic": "gpu"}
    ),
    Document(
        page_content="Dell partners with Nvidia to accelerate AI adoption in telecoms",
        metadata={"company":"Dell", "topic": "partnership"}
    ),
]

vector_store.add_documents(documents=documents, ids=ids)

results = vector_store.similarity_search_with_score(query="gpu", k=2)

for doc, score in results:
    print(f"Doc page cotent:\n {doc.page_content}\n\n \t[Score={score:.2f}]\n\t [Metadata={doc.metadata}]\n\n")