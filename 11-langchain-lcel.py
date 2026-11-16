"""
    Exercise 11 - LangChain LCEL

    Challenge:  None, just checking out LCEL
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel
from langchain_core.tracers.context import collect_runs
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

# Chaining invocations

prompt = PromptTemplate(template="Tellme everything about the anime character {character_name}")

parser = StrOutputParser()

parser.invoke(
    llm.invoke(
        prompt.invoke({"character_name": "Inosuke"})
    )
)

print_spacing()
# Runnables
# They can be:
#   - executed:
#       - invoke()
#       - batch()
#       - stream()
#   - inspected
#   - composed

runnables = [prompt, llm, parser]

# executed
print("-- Executed --")
for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINVOKE: {repr(runnable.invoke)}")
    print(f"\tBATCH: {repr(runnable.batch)}")
    print(f"\tSTREAM: {repr(runnable.stream)}")
    
# inspected
print("-- Inspected --")
for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINVOKE: {repr(runnable.get_input_schema())}")
    print(f"\tBATCH: {repr(runnable.get_output_schema())}")
    print(f"\tSTREAM: {repr(runnable.config_schema())}")
    
# config
print("-- Config --")
with collect_runs() as run_collection:
    result = llm.invoke(
        "Hello",
        config={
            "run_name": "demo_run",
            "tags": ["demo", "lcel"],
            "metadata": { "lesson": 2 }
        }
    )
    
    print(run_collection.traced_runs)
    print(run_collection.traced_runs[0].model_dump())

print_spacing()
# Compose Runnables

chain = RunnableSequence(prompt, llm, parser)

print(type(chain))

print(chain.invoke({"character_name": "Bartolomeo"}))

for chunk in chain.stream({"character_name": "Spike"}):
    print(chunk, end="", flush=True)
    
print(chain.batch([
    {"character_name": "Escanor"},
    {"character_name": "Deadpool"},
    {"character_name": "David Martinez"},
]))

chain.get_graph().print_ascii()

print_spacing()
# Function to runnable

def double(x: float) -> float:
    return 2*x

runnable = RunnableLambda(double)
runnable.invoke(4.0)

print_spacing()
# Parallel Runnables

parallel_chain = RunnableParallel(
    double=RunnableLambda(lambda x: x * 2),
    triple=RunnableLambda(lambda x: x * 3),
)

print(parallel_chain.invoke(8))

parallel_chain.get_graph().print_ascii()

print_spacing()
# LCEL

print("Chain: ",prompt | llm | parser )

chain = prompt | llm | parser

print("Chain invoke: ",chain.invoke(
    {"character_name": "Edward Elric"},
))