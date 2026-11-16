"""
    Exercise 14 - LangChain Tooling

    Challenge:  None, just checking out tooling
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

# Tool creation

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two given numbers.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: Multiplication between two numbers
    """
    result = x * y
    logger.info(f"Calculation: {x}*{y}={result}")
    return result

tools = [multiply]
tool_map = { tool.name: tool for tool in tools }

# Biding tools

llm_with_tools = llm.bind_tools(tools)

question = "What's 3 multiplied by 2?"

messages = [
    SystemMessage("You're a math assistant that should always rely on your tools to solve equations and only them. If you don't know the answer of a question just say \"I don't know\"."),
    HumanMessage(question)
]

assistant_message = llm_with_tools.invoke(messages)
print(assistant_message)

messages.append(assistant_message)

print_spacing()
# Using tool calls

parsed_tool_calls = assistant_message.tool_calls

print("Parsed tool calls: ", parsed_tool_calls)


for tool_call in parsed_tool_calls:
    tool_call_id = tool_call["id"]
    function_name = tool_call["name"]
    arguments = tool_call["args"]
    func = tool_map[function_name]
    result = func.invoke(arguments)
    tool_message = ToolMessage(
        content=result,
        name=function_name,
        tool_call_id=tool_call_id,
    )
    messages.append(tool_message)
  
print_spacing()  
# Send the result back

assistant_message = llm_with_tools.invoke(messages)

print(assistant_message)