"""
    Exercise 08 - Langchain Chatbot

    Challenge:  You're developing a virtual assistant for a company. Your chatbot needs to:
                    - Maintain conversation history.
                    - Respond consistently using predefined few-shot examples.
                    - Be customizable for different roles, such as:
                        - A robotic assistant with a sci-fi tone.
                        - A casual chatbot for fun interactions.
                        - A professional AI assistant for business tasks.

                At the end of this exercise, you’ll have a fully functional chatbot that can chat dynamically while following a predefined personality.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

load_dotenv()


class ChatBot:
    def __init__(
        self,
        examples: list[dict],
        name: str = "Franky",
        instructions: str = "You are a helpful assistant to help the user with general queries.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self._name = name
        self._instructions = instructions
        self._examples = examples
        self._model = model
        self._temperature = temperature

        self._llm = ChatOpenAI(model=self._model, temperature=self._temperature)

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self._examples,
        )

        self._messages = [
            SystemMessage(f"Your name is {name}"),
            SystemMessage(instructions),
            *prompt_template.invoke({}).to_messages()
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def examples(self) -> list[dict]:
        return self._examples

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    def invoke(self, message: str) -> AIMessage:
        self._messages.append(HumanMessage(message))
        assistant_response = self._llm.invoke(self._messages)
        self._messages.append(assistant_response)
        return assistant_response


examples = [
    {
        "input": "Hello!",
        "output": "SUUUUUUPEERR! How is the search for the One Piece going there?",
    },
    {
        "input": "What is the best ship ever?",
        "output": "Going Merry definitely, SUPEEEEERRRR cool!",
    },
]

chatbot = ChatBot(examples=examples)

print(chatbot.invoke("What's your favorite pirate crew?"))
