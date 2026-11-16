"""
    Exercise 15 - LangChain Agent Implementation

    Challenge:  None, just checking out agents in LangChain
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langchain_core.tools import tool
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

# Agent class

class Agent:
    def __init__(
        self,
        # examples: list[dict],
        name: str = "Franky",
        role: str = "Shipwright",
        instructions: str = "You are a shipwright for the Straw Hat pirates. Luffy's crew.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: list[StructuredTool] | None = None
    ):
        self._name = name
        self._instructions = instructions
        self._role = role
        self._model = model
        self._temperature = temperature
        self._tools = tools
        self._tool_map = { tool.name: tool for tool in self._tools}

        self._llm = ChatOpenAI(model=self._model, temperature=self._temperature)

        self._memory = [
            SystemMessage(f"Your name is {name}"),
            SystemMessage(instructions),
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def role(self) -> str:
        return self._role

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def tools(self) -> list[StructuredTool] | None:
        return self._tools

    @property
    def tool_map(self) -> dict[str, StructuredTool]:
        return self._tool_map

    def _invoke_llm(self):
        llm = self._llm.bind_tools(self._tools)
        response = llm.invoke(self._memory)
        self._memory.append(response)
        return response
    
    def _call_tools(self, tool_calls: list[dict]):
        for tool_call in tool_calls:
            func = self._tool_map[tool_call["name"]]
            result = func.invoke(tool_call["args"])
            self._memory.append(
                ToolMessage(content=result, name=tool_call["name"], tool_call_id=tool_call["id"])
            )

    def invoke(self, message: str) -> AIMessage:
        self._memory.append(HumanMessage(message))
        assistant_response = self._invoke_llm()
        
        if assistant_response.tool_calls:
            self._call_tools(assistant_response.tool_calls)
            assistant_response = self._invoke_llm()
            
        return assistant_response
    
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

franky_the_agent = Agent(tools=[multiply])

print(franky_the_agent.invoke("What's 3 multiplied by 2?").content)