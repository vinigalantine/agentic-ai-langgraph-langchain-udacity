"""
    Exercise 04 - Tool Calling

    Challenge: Instead of manually deciding when to call which function, your AI agent will automatically detect when a tool is needed and invoke it.
"""

from openai import OpenAI, NOT_GIVEN
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
import json
from dotenv import load_dotenv
import inspect
import datetime
from collections.abc import Callable, Mapping
from typing import get_type_hints, Any
from types import MappingProxyType
import logging

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Tool:
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = inspect.getdoc(func)
        self.argument_types_map = get_type_hints(func)
        self.signature = inspect.signature(func)
        self.arguments = [
            {
                "name": key,
                "type": self._infer_json_schema_type(value),
                "required": param.default == inspect.Parameter.empty
            }
            for key, value in self.argument_types_map.items()
            if (param := self.signature.parameters.get(key))
        ]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        argument["name"]: {
                            "type": argument["type"],
                        }
                        for argument in self.arguments
                    },
                    "required": [
                        argument["name"]
                        for argument in self.arguments
                        if argument["required"]
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)
    
    def _infer_json_schema_type(self, arg_type: Any) -> str:
        if arg_type == bool:
            return "boolean"
        elif arg_type == int:
            return "integer"
        elif arg_type == float:
            return "number"
        elif arg_type == str:
            return "string"
        elif arg_type == list:
            return "array"
        elif arg_type == dict:
            return "object"
        elif arg_type is type(None):
            return "null"
        elif arg_type == datetime.date or arg_type == datetime.datetime:
            return "string"
        else:
            return "string"

def power(base: float, exponent: float) -> float:
    """Calculate a number raised to a given power.
    
        Args:
            base (float): The number to exponentiate
            exponent (float): The exponent
    
        Returns:
            float: The result of the exponentiation
    """
    
    result = base ** exponent
    logger.info(f"Calculation: {base}^{exponent}={result}")
    return result  

def get_weather(city: str) -> dict | str:
    """Get weather from city

    Args:
        city (str): City for which you want to retrieve the weather

    Returns:
        dict | str: A dict with the "temperature" and "condition" or a message if the city is not on the list.
    """
    logger.info(f'Information from city "{city}" requested.')
    
    # Using the data from my "mcp-learning" repo
    fake_weather = {
        "são paulo": {"temperature": 20.0, "condition": "Cloudy"},
        "tokyo": {"temperature": 26.0, "condition": "Sunny"},
        "washington": {"temperature": 6.2, "condition": "Snowy"},
        "london": {"temperature": 12.1, "condition": "Foggy"},
        "dublin": {"temperature": 7.5, "condition": "Rainy"},
    }
    
    return fake_weather.get(city.lower(), "Weather data not available.")

power_tool = Tool(power)
weather_tool = Tool(get_weather)

TOOLS = [power_tool, weather_tool]

class Agent:
    def __init__(
        self,
        name: str = "Franky",
        role: str = "General Assistant",
        instructions: str = "Help the user with general queries.",
        tools: list[Tool] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI()  # Would be good to be "client-agnostic" but for practical reasons let's keep it OpenAI

        try:
            available_models = [
                openai_model.id for openai_model in self._client.models.list()
            ]
        except Exception:
            available_models = ["gpt-4o-mini", "gpt-4o"]

        if model not in available_models:
            raise ValueError(
                f'Invalid model: "{model}". Please provide a valid OpenAI model.'
            )

        self._name = name
        self._role = role
        self._instructions = instructions
        self._tools = list(tools or [])
        self._tools_map = {tool.name: tool for tool in self._tools}
        self._openai_tools = [tool.to_schema() for tool in self._tools] if self._tools else NOT_GIVEN

        self.model = model
        self.temperature = temperature

        self._messages = [
            {
                "role": "system",
                "content": f'Your name is {self._name} and you are a "{self._role}" assistant.',
            },
            {"role": "system", "content": self._instructions},
        ]
        
    def _get_message(self, messages: list[dict]) -> ChatCompletionMessage:
        response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=self._openai_tools
            )
        
        return response.choices[0].message
    
    def _call_tools(self, tool_calls: list[ChatCompletionMessageToolCall], loop_count: int = 0) -> None:
        if loop_count > 5:
            self._messages.extend({"role": "tool", "content": "Error: tool call limit reached", "tool_call_id": tool_call.id} for tool_call in tool_calls)
            self._messages.append({"role": "assistant", "content": "I hit the tool call limit and stopped."})
            return
        
        messages_to_add = []
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            function_name = tool_call.function.name
            callable_tool = self._tools_map.get(function_name)
            
            if callable_tool is None:
                result = f"Error: unknown tool '{function_name}'"
            else:
                try: 
                    args = json.loads(tool_call.function.arguments)
                    result = callable_tool(**args) 
                except Exception as e: 
                    result = f"Error: {e}"
            
            messages_to_add.append({"role": "tool", "content": str(result), "tool_call_id": tool_call_id})
        
        try:
            assistant_message = self._get_message(self._messages + messages_to_add)
        except Exception as e:
            raise RuntimeError("Failed to get the agent response") from e
        
        self._messages.extend(messages_to_add)
        
        tool_calls = assistant_message.tool_calls
        self._messages.append({
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            **({"tool_calls": [tool_call.model_dump() for tool_call in tool_calls]} if tool_calls else {})
                          })
        
        if tool_calls:
            self._call_tools(tool_calls, loop_count+1)
        
    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def messages(self) -> tuple[dict, ...]:
        return tuple(self._messages)
    
    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools)
    
    @property
    def tools_map(self) -> Mapping[str, Tool]:
        return MappingProxyType(self._tools_map)
    
    @property
    def openai_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._openai_tools or ())

    def invoke(self, message: str) -> str:
        message_chat = {"role": "user", "content": message}

        try:
            assistant_message = self._get_message(self._messages + [message_chat])
        except Exception as e:
            raise RuntimeError("Failed to get the agent response") from e

        self._messages.append(message_chat)
        tool_calls = assistant_message.tool_calls
        self._messages.append({
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            **({"tool_calls": [tool_call.model_dump() for tool_call in tool_calls]} if tool_calls else {})
                          })
        
        if tool_calls:
            self._call_tools(tool_calls)

        return self._messages[-1]["content"] or "[Agent did not respond]"

agent = Agent(tools=TOOLS)
print(agent.invoke("What is the weather in Tokyo?"))
print(agent.invoke("What is 2 to the power of 10?"))
print(agent.invoke("Tell me the weather in São Paulo and then the result of 2 squared."))