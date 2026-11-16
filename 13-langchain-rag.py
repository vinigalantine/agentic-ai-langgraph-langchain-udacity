"""
    Exercise 13 - LangChain RAG

    Challenge:  None, just checking out RAG
"""

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

def print_spacing():
    print("\n" + "=" * 50 + "\n")
    
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Loader
loader = WikipediaLoader(
    "Franky_(One_Piece)",
    load_max_docs=1,
    doc_content_chars_max=40000
)

docs = loader.load()

print(f"Number of docs: {len(docs)}")
print(f"Doc page content len: {len(docs[0].page_content)}")
print(f"Doc page content: \n", docs[0].page_content)

print_spacing()
# Splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs_splitted = text_splitter.split_documents(docs)

print(f"Split Wikipedia into {len(docs_splitted)} sub-docs")
print("A banana split: ", docs_splitted[0])

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Store
vector_store = InMemoryVectorStore(embeddings)

document_ids = vector_store.add_documents(documents=docs_splitted)
print("Some ids for you: ", document_ids[:10])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print_spacing()
# Prompt

# Ngl copied from the course
template = ChatPromptTemplate([
    ("system", "You are an assistant for question-answering tasks."),
    ("human", "Use the following pieces of retrieved context to answer the question. "
              "If you don't know the answer, just say that you don't know. " 
              "Use three sentences maximum and keep the answer concise. "
              "\n# Question: \n-> {question} "
              "\n# Context: \n-> {context} "
              "\n# Answer: ")
])

def format_docs(docs):
    return "\n\n-> ".join(doc.page_content for doc in docs)
    
# Generation
question = "When was Franky first appearance on the anime?"
context = format_docs(retriever.invoke(question))
messages = template.invoke({ "question": question, "context": context }).to_messages()
answer = llm.invoke(messages)

print("Answer: ", answer.content)

print_spacing()
# LCEL

rag_chain = (
                RunnableParallel(context= retriever | format_docs, question= RunnablePassthrough())
                | template
                | llm
            )

print(rag_chain.invoke("When was Franky first appearance on the manga?"))