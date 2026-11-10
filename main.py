from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= [
        { "role": "system", "content": "You are a Pokémon master." },
        { "role": "user", "content": "If you could choose any starter from any generation, which one would it be?" },
    ]
)

print(response.choices[0].message.content)