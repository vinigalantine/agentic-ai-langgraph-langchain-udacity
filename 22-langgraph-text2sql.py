"""
    Exercise 22 - LangGraph Text2SQL

    Challenge:  You’re building a Text2SQL assistant for a Sales Dashboard. The agent should:

                    - Parse user questions.
                    - Identify the relevant tables and columns.
                    - Generate the corresponding SQL query.
                    - Execute the query and return the result.
                
                This enables non-technical users to ask questions in plain English and get data insights without writing SQL.
"""

from dotenv import load_dotenv
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# The exercise imports the tools from a separate file, and I could've reused exercise 21 for that,
# but I feel it's better to have them here, from the POV of how I learn best.
@tool
def list_tables(config: RunnableConfig) -> list[str]:
    """List all tables in the database

    Args:
        config (RunnableConfig): Expects the database configuration.

    Returns:
        list[str]: List of the table names
    """
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)
    
    logger.info("List of tables retrieved.")
    
    return inspector.get_table_names()

@tool
def get_table_schema(table_name: str, config: RunnableConfig) -> list[dict]:
    """Get schema information about a table.

    Args:
        table_name (str): Table name
        config (RunnableConfig): Expects the database configuration.

    Returns:
        list[dict]: Table schema
    """
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)
    logger.info(f"Table {table_name} desc retrieved.")
    
    return inspector.get_columns(table_name)

@tool
def execute_sql(query: str, config: RunnableConfig) -> list[tuple]:
    """Execute a SQL query and return the result of it.

    Args:
        query (str): Query to be executed
        config (RunnableConfig): Expects the database configuration.

    Returns:
        list[tuple]: Result of the query
    """
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    with db_engine.begin() as connection:
        result = connection.execute(text(query)).fetchall()
        
    logger.info(f'Query executed "{query}"')
    
    return result

class State(MessagesState):
    user_query: str

def messages_builder(state: State):
    # Copied this from the exercise
    system_message = (
        "You are a Sr. SQL developer tasked with generating SQL queries. Perform the following steps:\n"
        "First, find out the appropriate table name based on all tables. "
        "Then get the table's schema to understand the columns. "
        "With the table name and the schema, generate the ANSI SQL query you think is applicable to the user question. "
        "Finally, use a tool to execute the above SQL query and output the result based on the user question."
    )
    messages = [
        SystemMessage(system_message),
        HumanMessage(state["user_query"])
    ]
    return { "messages" : messages }

def db_agent(state: State):
    agent_message = db_llm.invoke(state["messages"])
    agent_message.name = "bill_nye_the_dba_guy"
    
    return { "messages": agent_message }

def should_continue(state: State):
    messages = state["messages"]
    
    if messages[-1].tool_calls:
        return "db_tools"
    
    return END

db_tools = [list_tables, get_table_schema, execute_sql]
db_llm = llm.bind_tools(db_tools, tool_choice="auto")

workflow = StateGraph(State)
workflow.add_node("db_tools", ToolNode(db_tools))
workflow.add_node("messages_builder", messages_builder)
workflow.add_node("db_agent", db_agent)

workflow.add_edge(START, "messages_builder")
workflow.add_edge("messages_builder", "db_agent")
workflow.add_conditional_edges(
                               source="db_agent",
                               path=should_continue,
                               path_map=["db_tools", END]
                               )
workflow.add_edge("db_tools", "db_agent")

graph = workflow.compile()

print("Graph: ")
graph.get_graph().print_ascii()

db_engine = create_engine("sqlite:///data/sales.db")

config = { "configurable": { "db_engine": db_engine } }
inputs = { "user_query": "How many Dell XPS 15 were sold?" }

response = graph.invoke(
    input=inputs,
    config=config
)

for message in response["messages"]:
    message.pretty_print()