"""
    Exercise 07 - LangChain
    
    Challenge: None, just basic langchain stuff.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def print_spacing():
    print("\n"+"="*50+"\n")

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

print("Simple invoke: ", llm.invoke("Hello there").content)

print_spacing()
# Chat example

print("Chat example:", llm.invoke([
        SystemMessage("You are a fan of the anime One Piece."),
        HumanMessage("What's the name of Luffy's first ship?")
    ]).content)

print_spacing()
# Prompt Template

prompt_template = PromptTemplate(template="Tell me everything about the music {music_name}")

print("Prompt template format: ", prompt_template.format(music_name="Come Undone"))

print("Prompt template chat: ", llm.invoke(prompt_template.invoke({"music_name":"Come Undone"})).content)

print_spacing()
# Few Shot Prompt

example_prompt = PromptTemplate(template="{character_name}: I'm about to use my {power}, to {action}.")
examples = [
    {
        "character_name": "Luffy",
        "power": "Gomu Gomu no Pistol",
        "action": "get the One Piece."
    },
    {
        "character_name": "Jonathan Joestar",
        "power": "Hamon",
        "action": "kill Dio."
    },
    {
        "character_name": "Inosuke",
        "power": "Beast Breathing",
        "action": "headbutt that tree."
    }
]

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="{character_name}:",
    input_variables=["character_name"]
)

print("Few shot example:", llm.invoke(prompt_template.invoke({"character_name": "Zoro"})).content )