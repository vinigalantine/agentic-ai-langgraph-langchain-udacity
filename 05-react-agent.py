"""
    Exercise 05 - ReAct agent

    Challenge: Implement an Agent with the ReAct pattern
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

def add(x: float, y: float) -> float:
    """Add two numbers.
    
        Args:
            x (float): First number
            y (float): Second number
    
        Returns:
            float: The result of the sum.
    """
    
    result = x + y
    logger.info(f"Calculation: {x}+{y}={result}")
    return result 

class StopReActLoopException(Exception):
    """Raised to terminate the ReAct loop"""
    
TERMINATION_MESSAGE = "StopReActLoopException"

def terminate() -> str:
    """Terminate the ReAct loop when the agent reaches the conclusion there are no more actions to take.

    Returns:
        str: Termination message
    """
    return TERMINATION_MESSAGE

power_tool = Tool(power)
add_tool = Tool(add)
terminate_tool = Tool(terminate)

TOOLS = [power_tool, add_tool, terminate_tool]

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

        # Copied the instructions from the exercise.
        self._messages = [
            {
                "role": "system",
                "content": f'Your name is {self._name} and you are a "{self._role}" assistant.',
            },
            {"role": "system", "content": self._instructions +
                                          "You can answer multistep questions by sequentially calling functions. "
                                          "You follow a pattern of Thought and Action. "
                                          "Create a plan of execution: "
                                          "- Use Thought to describe your thoughts about the question you have been asked. "
                                          "- Use Action to specify one of the tools available to you. "
                                          "When you think it's over call the terminate tool. "
                                          "Never try to respond directly if the question needs a tool."
            },
        ]
        
    def _get_message(self, messages: list[dict], tool_choice: Any = NOT_GIVEN) -> ChatCompletionMessage:
        response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=self._openai_tools,
                tool_choice=tool_choice
            )
        
        return response.choices[0].message
    
    def _reason(self) -> None:
        assistant_message = self._get_message(messages=self._messages, tool_choice="none")
        
        self._messages.append({
                                "role": "assistant",
                                "content": assistant_message.content or ""
                              })
        
    def _react_loop(self, max_iter: int) -> None:
        for i in range(max_iter):
            self._reason()
            
            logger.info(f"ReAct loop iteration: {i}")
            
            assistant_message = self._get_message(messages=self._messages)
            tool_calls = assistant_message.tool_calls
            
            self._messages.append({
                                    "role": "assistant",
                                    "content": assistant_message.content or "",
                                    **({"tool_calls": [tool_call.model_dump() for tool_call in tool_calls]} if tool_calls else {})
                                  })
                    
            if tool_calls:
                self._call_tools(tool_calls)
            
    
    def _call_tools(self, tool_calls: list[ChatCompletionMessageToolCall]) -> None:
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
            
            self._messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call_id})
            
            if result == TERMINATION_MESSAGE:
                raise StopReActLoopException(TERMINATION_MESSAGE)
        
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

    def invoke(self, message: str, max_iter: int = 3) -> str:
        if not 1 <= max_iter <= 5:
            raise ValueError('The variable "max_iter" must be between 1 and 5.')
        
        self._messages.append({"role": "user", "content": message})

        try:
            self._react_loop(max_iter=max_iter)
        except StopReActLoopException as e:
            logger.info(f'Terminated: "{e}"')

        for msg in reversed(self._messages):
            if msg["role"] == "assistant" and msg["content"]:
                return msg["content"]

        return "[Agent did not respond]"
    
franky = Agent(tools=TOOLS)
print(franky.invoke("What is 2 squared, plus 5?", 5))