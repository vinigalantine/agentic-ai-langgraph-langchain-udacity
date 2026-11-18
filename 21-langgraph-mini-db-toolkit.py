"""
    Exercise 21 - LangGraph Mini DB Toolkit

    Challenge:  Base for the next exercise
"""

import logging
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text, create_engine
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

db_engine = create_engine("sqlite:///data/sales.db")

inspector = sqlalchemy.inspect(db_engine)

print("Tables: ", inspector.get_table_names())

table_name = "sales"
schema = inspector.get_columns(table_name)

print("Columns desc: ", schema)

column_names = [column["name"] for column in schema]

print("Columns names: ", column_names)

sql = f"SELECT * FROM {table_name} LIMIT 10"

with db_engine.begin() as connection:
    result = connection.execute(text(sql)).fetchall()
    
print("SQL Result: ", result)

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

print_spacing()

config = { "configurable": { "db_engine": db_engine } }
tables = list_tables.invoke({}, config)
sql = f"SELECT * FROM {table_name} LIMIT 10"

schema = {
    table: get_table_schema.invoke({
        "table_name": table
    }, config) for table in tables
}
print("Schema: ", schema)

print_spacing()

print(execute_sql.invoke({"query": sql}, config))