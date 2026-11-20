# Udacity - AI Agents with LangChain and LangGraph

My exercises for the Udacity course [AI Agents with LangChain and LangGraph](https://www.udacity.com/course/ai-agents-with-langchain-and-langgraph--cd13764).

I had already worked a bit with both technologies; this repo is me getting deeper into them. It goes from raw OpenAI SDK calls and hand-made agents, to LangChain, to LangGraph workflows, and ends with RAG, human-in-the-loop, observability (MLflow) and evaluation (Ragas).

## Setup

This project uses `venv` (if you don't know what it is, don't ask me... jk, here: [What is venv?](https://docs.python.org/3/library/venv.html)).

### 1. Create the `venv`

```shell
python -m venv .venv
```

### 2. Activate it

Cmd
```bat
.venv\Scripts\activate.bat
```

PowerShell
```shell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

Run this on a brand new `venv` or whenever `requirements.txt` changes (with the `venv` activated):

```shell
pip install -r requirements.txt
```

### 4. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

- `OPENAI_API_KEY` - required by almost every exercise.
- `OPENAI_BASE_URL` - only if you use a custom OpenAI-compatible endpoint.
- `TAVILY_API_KEY` - for the web search exercises (27, 32, 33, 34).

### 5. Running an exercise

Each exercise is a standalone script:

```shell
python 01-simple-llm-call.py
```

### Extra requirements for some exercises

- **MLflow server (34 and 36):** start it in a separate terminal before running the exercise:

  ```shell
  mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root ./mlartifacts
  ```

  The UI will be available at http://127.0.0.1:5000.

- **PDF (32 and 36):** these exercises read `data/compact-guide-to-large-language-models.pdf` (Databricks' "Compact Guide to Large Language Models"). The file is not included in the repo; download it and place it in `data/`.

## Exercises

### Foundations (OpenAI SDK, no frameworks)

1. **[01-simple-llm-call.py](01-simple-llm-call.py)** - A single LLM call with the OpenAI SDK, using a system prompt and temperature to build a small content assistant for marketing analysts. Takes the query from the terminal.
2. **[02-create-agent-from-scratch.py](02-create-agent-from-scratch.py)** - A reusable `Agent` class with a role, instructions and message history. Creates several agents (default, travel agent, math tutor, storyteller) with different settings.
3. **[03-adding-self-reflection-capability.py](03-adding-self-reflection-capability.py)** - Adds self-reflection to the agent: it critiques and improves its own answer for up to 3 iterations.
4. **[04-tool-calling.py](04-tool-calling.py)** - A `Tool` class that turns Python functions into OpenAI tool schemas, so the agent decides by itself when to call them (`power`, fake `get_weather`).
5. **[05-react-agent.py](05-react-agent.py)** - The ReAct pattern (Thought → Action → Observation) with a `terminate` tool to end the loop.
6. **[06-multi-agent.py](06-multi-agent.py)** - A main agent that delegates work to specialist sub-agents, exposed to it as tools.

### LangChain

7. **[07-langchain.py](07-langchain.py)** - LangChain basics: `ChatOpenAI`, message types, prompt templates and few-shot prompts.
8. **[08-langchain-chatbot.py](08-langchain-chatbot.py)** - A `ChatBot` class with conversation history, few-shot examples and customizable personalities (sci-fi robot, casual, professional).
9. **[09-langchain-stream.py](09-langchain-stream.py)** - Streaming tokens with `stream()` and `astream_events()`, accumulating chunks and handling interruptions.
10. **[10-langchain-structured-outputs.py](10-langchain-structured-outputs.py)** - Output parsers and `with_structured_output()` with Pydantic and TypedDict schemas, including error handling.
11. **[11-langchain-lcel.py](11-langchain-lcel.py)** - LCEL: composing runnables with `|`, `RunnableLambda`, `RunnableParallel`, batching, streaming, schemas and run config.
12. **[12-langchain-multi-step-workflows.py](12-langchain-multi-step-workflows.py)** - An "AI Business Advisor" chain: generates a business idea for an industry, analyzes it and returns a structured report.
13. **[13-langchain-rag.py](13-langchain-rag.py)** - A first RAG pipeline: loads Wikipedia pages, splits, embeds them into an in-memory vector store and answers questions from the retrieved context.
14. **[14-langchain-tooling.py](14-langchain-tooling.py)** - Tools with the `@tool` decorator, `bind_tools()`, reading `tool_calls` and returning `ToolMessage` results.
15. **[15-langchain-agent-implementation.py](15-langchain-agent-implementation.py)** - An `Agent` class in LangChain that loops between the LLM and its tools until it has an answer.

### LangGraph

16. **[16-langgraph-workflows.py](16-langgraph-workflows.py)** - First `StateGraph` workflows: a simple sequential graph and one with LLM calls using `MessagesState`.
17. **[17-langgraph-router.py](17-langgraph-router.py)** - Conditional edges: routes a string to "reverse", "upper" or an error node based on the requested action.
18. **[18-langgraph-state-management.py](18-langgraph-state-management.py)** - State management, including a traffic light simulation that loops with conditional edges.
19. **[19-langgraph-reducers.py](19-langgraph-reducers.py)** - Reducers (`operator.add`, `add_messages`) to merge state updates from parallel nodes.
20. **[20-langgraph-configurable-parameters.py](20-langgraph-configurable-parameters.py)** - Passing runtime parameters to nodes through `RunnableConfig`.
21. **[21-langgraph-mini-db-toolkit.py](21-langgraph-mini-db-toolkit.py)** - Database tools (`list_tables`, `get_table_schema`, `execute_sql`) over `data/sales.db`, with the DB engine injected through config. Base for exercise 22.
22. **[22-langgraph-text2sql.py](22-langgraph-text2sql.py)** - A Text2SQL agent: answers plain-English questions about sales by exploring the schema, writing SQL and running it.
23. **[23-langgraph-limiting-messages.py](23-langgraph-limiting-messages.py)** - Keeping the context small with `RemoveMessage` and `trim_messages()`, comparing token usage. Also introduces checkpointers.
24. **[24-langgraph-multiple-schemas.py](24-langgraph-multiple-schemas.py)** - Separate input, output and internal state schemas in a ticket-processing graph.
25. **[25-langgraph-checkpoints.py](25-langgraph-checkpoints.py)** - Saving and inspecting state with `MemorySaver`, `thread_id` and `get_state_history()`.
26. **[26-langgraph-loan-agent.py](26-langgraph-loan-agent.py)** - A loan agent that calculates the maximum loan for a customer, tracks the negotiation status (won/lost) via `Command` and summarizes long conversations.

### LangGraph + external services, memory and RAG

27. **[27-langchain-langgraph-calling-apis.py](27-langchain-langgraph-calling-apis.py)** - An agent calling external APIs as tools: the Rick and Morty API and Tavily web search.
28. **[28-langgraph-long-term-memory.py](28-langgraph-long-term-memory.py)** - Persistent memory with `SqliteSaver` (`data/memory.db`) compared to in-memory checkpoints.
29. **[29-langgraph-langmem.py](29-langgraph-langmem.py)** - Semantic long-term memory with LangMem: the agent saves and searches memories across threads.
30. **[30-langchain-embeddings.py](30-langchain-embeddings.py)** - Embeddings with OpenAI and HuggingFace, similarity scores and a 2D PCA plot saved to `data/embeddings.png`.
31. **[31-langchain-chromadb.py](31-langchain-chromadb.py)** - ChromaDB: collections, queries, different embedding functions, and LangChain's `Chroma` wrapper, persisted in `data/chromadb/`.
32. **[32-langchain-langgraph-agentic-rag.py](32-langchain-langgraph-agentic-rag.py)** - Agentic RAG: retrieves from a PDF, an LLM decides whether the context is good enough, and if not, falls back to a web search agent.

### Human-in-the-loop, observability and evaluation

33. **[33-langchain-langgraph-human-in-the-loop.py](33-langchain-langgraph-human-in-the-loop.py)** - Human-in-the-loop with `interrupt_before`: the user approves tool calls or edits the question before the agent continues.
34. **[34-langchain-langgraph-mlflow-observability.py](34-langchain-langgraph-mlflow-observability.py)** - Tracing a LangGraph agent with MLflow (`@mlflow.trace` and `mlflow.langchain.autolog()`).
35. **[35-langchain-langgraph-evaluation.py](35-langchain-langgraph-evaluation.py)** - Evaluating a RAG chain (context recall, faithfulness, factual correctness) and an agent (tool call accuracy, goal accuracy) with Ragas.
36. **[36-langchain-langgraph-evaluate-agent.py](36-langchain-langgraph-evaluate-agent.py)** - Final exercise: a RAG workflow over a PDF that evaluates its own answers with an LLM judge and Ragas metrics, and logs the parameters and scores to MLflow.

## Disclosure and versions

- **AI usage:** this README was generated with AI (GitHub Copilot) based on the code in the repo. The exercises were written by me, with a few exceptions noted in their own docstrings: the approaches in 23 and 24 came from Claude, and the starter for 36 was rewritten with Claude because the original was outdated. I also used AI to review my solutions.
- **Main versions** (full list in [requirements.txt](requirements.txt)):

  | Package | Version |
  |---|---|
  | Python | 3.12.4 |
  | langchain | 1.5.2 |
  | langchain-core | 1.8.3 |
  | langchain-openai | 1.1.9 |
  | langgraph | 1.2.22 |
  | openai | 1.109.1 |
  | chromadb | 1.5.9 |
  | mlflow | 3.18.0 |
  | ragas | 0.4.4 |

- Some of the course's starter code uses older APIs (e.g. `langchain_community` loaders, legacy Ragas metrics). A few exercises (e.g. 32, 35, 36) were updated to the versions above, so they may differ from the course's original code.
