"""
    Exercise 10 - LangChain Structured Outputs

    Challenge:  None, just checking out parsers
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict
import asyncio

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

# String parser

parser = StrOutputParser()
message = "Name Luffy's crew in one line."

print("Raw message: ", repr(llm.invoke(message)))
print("Parsed: ", parser.invoke(llm.invoke(message)))

print_spacing()
# Same but in a chain

prompt = ChatPromptTemplate.from_template("Name {character}'s crew in one line.")
chain = prompt | llm | parser

print("Chain: ", chain.invoke({"character": "Bartolomeo"}))

print_spacing()
# TypedDict schema

class CharacterInfo(TypedDict):
    """Anime character basic info."""
    name: str
    anime: str

llm_with_dict = llm.with_structured_output(CharacterInfo)

print("TypedDict: ", llm_with_dict.invoke("Who is Jotaro?"))

print_spacing()
# Pydantic schema

class Stand(BaseModel):
    """JoJo's Bizarre Adventure stand."""
    
    name: str = Field(description="The stand's name")
    user: str = Field(description="The character who wields it")
    abilities: list[str] = Field(description="Abilities that the stand is known for")
    part: int | None = Field(default=None, description="Part number it debuted")
    
    
llm_with_stand = llm.with_structured_output(Stand)
star_platinum = llm_with_stand.invoke("Describe Star Platinum")

print("Pydantic: ", star_platinum)
print("Attribute access: ", star_platinum.user, "->", star_platinum.abilities[0])

print_spacing()
# Parsing raw text and error

parser = PydanticOutputParser(pydantic_object=Stand)

print("Round trip:", parser.parse(star_platinum.model_dump_json()))

malformed = "{ 'name': 'The World', 'user': 'Dio Brando' }"

try:
    parser.parse(malformed)
except OutputParserException as error:
    print("Failed as expected: ", str(error).splitlines()[0])
    
    repair_prompt = (
        "Fix this so it matches the schema. Output JSON only. \n"
        f"Schema:\n{parser.get_format_instructions()}\n"
        f"Broken:\n{malformed}"
    )
    
    print("Repaired:", parser.parse(llm.invoke(repair_prompt).content))
    
print_spacing()
# Async batch

async def describe_many(names: list[str]) -> list[Stand]:
    return await llm_with_stand.abatch([f"Describe {name}" for name in names])

for stand in asyncio.run(describe_many(["Crazy Diamond", "Gold Experience"])):
    print(f"{stand.user} -> {stand.name}: {stand.abilities[0]}")