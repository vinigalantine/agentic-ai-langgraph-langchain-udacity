"""
    Exercise 01 - Simple LLM Call
    
    Challenge: You are tasked with creating an application that helps Marketing Analysts with creating content.
    
    TODOS:
    TODO - Instantiate your client
    TODO - Create a variable called model with "gpt-4o-mini" as value
    TODO - Create a variable called temperature with some float value ranging from 0.0 to 2.0
    TODO - Create a variable called system_prompt with any instruction text.
    TODO - Create a function to reuse LLM calls
    TODO - Marketing analyst query
    TODO - Pass the other arguments
    TODO - Pass the other arguments
"""

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
SYSTEM_PROMPT = "You are an AI assistant designed to help Marketing Analysts create, refine, and optimize marketing content for different audiences, platforms, and objectives."

def create_content(query:str, model:str = MODEL, temperature:float = TEMPERATURE) -> str :
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages= [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": query },
        ]
    )
    
    return response.choices[0].message.content

print("What's your query?")
querry_input = input()

print(create_content(querry_input))
