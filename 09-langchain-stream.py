"""
    Exercise 09 - LangChain Stream

    Challenge:  None, just checking out "stream"

    Obs: line 93 clears the console comment it if needed
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    AIMessageChunk,
)
import asyncio

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
)


def print_spacing():
    print("\n" + "=" * 50 + "\n")


MESSAGE = "What is a stand in JoJo's Bizarre Adventure?"

chunks = []

print("Stream")
for chunk in llm.stream(MESSAGE):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

    if len(chunks) % 12 == 0:
        print("\n")

print_spacing()

# Message chunk
message_chunk = AIMessageChunk("")

for chunk in chunks:
    message_chunk = message_chunk + chunk

print("Message chunk: ", message_chunk)

print_spacing()

# Interrupt

chunks = []
memory = [MESSAGE]
# To properly test this one, pay attention when executing the code 😂
# This is better in a Jupyter Notebook, so we are simulating a KeyboardInterrupt
try:
    for chunk in llm.stream(MESSAGE):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

        if len(chunks) == 50:
            raise KeyboardInterrupt

        if len(chunks) % 12 == 0:
            print("\n")

except KeyboardInterrupt:
    print("\n" + "-" * 20 + "Interrupted" + "-" * 20)

result = "".join([chunk.content for chunk in chunks])
memory.append(AIMessage(content=result))

interrupted_message = 'If you think the message was interrupted, continue from the last word or if that was the end just output "SUUUPEERR!"'
memory.append(SystemMessage(content=interrupted_message))

for chunk in llm.stream(memory):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

    if len(chunks) % 12 == 0:
        print("\n")

print_spacing()
# Processing

chunks = []
word_count = 0

for chunk in llm.stream(MESSAGE):
    print("\033[H\033[J", end="")

    chunks.append(chunk)

    words = "".join([chunk.content for chunk in chunks])
    word_count = len(words.split())

    print(words)
    print(f"-- Word count: {word_count} --", end="\n")

    if len(chunks) % 12 == 0:
        print("\n")

print_spacing()
# Streaming Events


async def stream_events():
    events = []
    async for event in llm.astream_events("Hello there", version="v2"):
        if event["event"] == "on_chat_model_start":
            print("Start streaming...")
        if event["event"] == "on_chat_model_stream":
            print(f"Chunk: {repr(event['data']['chunk'].content)}")
        if event["event"] == "on_chat_model_end":
            print("--GRAND FINALE--")

        events.append(event)
        
    print(f"{len(events)} events.")


asyncio.run(stream_events())
