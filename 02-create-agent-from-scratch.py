"""
    Exercise 02 - Create Agent from scratch
    
    Challenge: You are tasked with creating an abstraction on top of LLMs, that is adaptable and efficient.
    
    TODOS:
    TODO - Instantiate your client
    TODO - Craft your prompts
    TODO - Create a Agent class
        TODO - Create your constructor with required parameters
        TODO - Instantiate your client properly
        TODO - Create the logic for your invoke() method
    TODO - Create a default agent without role and instructions
            Then ask it simple questions like:  
            "What is the capital of France?"
    TODO - Create a travel agent with role and instructions
            Then ask it questions like:
            Where should I go for vacation in December?
    TODO - Create a math tutor agent with role and instructions
            Then ask it questions like:
            How do I solve a quadratic equation?
    TODO - Create a creative storyteller agent with role and instructions
            Remember, the higher temperature the more creative it will be 
            Then ask it questions like:
            Tell me a story about a dragon and a wizard
"""

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, name:str="Franky", role:str="General Assistant", instructions:str="Help the user with general queries.", model:str = "gpt-4o-mini", temperature:float=0.0) -> None:
        self._client = OpenAI() # Would be good to be "client-agnostic" but for practical reasons let's keep it OpenAI
        
        try:
            available_models = [openai_model.id for openai_model in self._client.models.list()]
        except Exception:
            available_models = ["gpt-4o-mini", "gpt-4o"]
            
        if model not in available_models:
            raise ValueError(f'Invalid model: "{model}". Please provide a valid OpenAI model.')
        
        self._name = name
        self._role = role
        self._instructions = instructions

        self.model = model
        self.temperature = temperature
        
        self._messages = [{ "role": "system", "content": f'Your name is {self._name} and you are a "{self._role}" assistant.' },
                          { "role": "system", "content": self._instructions }]
        
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
    
    def invoke(self, message: str) -> str:
        message_chat = {"role": "user", "content": message }
        
        try:    
            response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=self._messages + [message_chat]
            )
        except Exception as e:
            raise RuntimeError("Failed to get the agent response") from e
        
        # After success it adds the user message to "history"
        self._messages.append(message_chat)
        
        # Getting and adding the agent message to the "history"
        agent_response = response.choices[0].message.content
        
        if agent_response:
            self._messages.append({"role": "assistant", "content": agent_response })
        
        return agent_response or "[Agent did not respond]"
    
def print_agent_response(agent: Agent, response: str) -> None:
    print(f"{agent.role} - {agent.name} the agent: {response}")

def print_spacing():
    print("\n" + "="*100 + "\n")

default_agent = Agent()
query = "What is the capital of France?"
agent_response = default_agent.invoke(query)
print_agent_response(default_agent, agent_response)

print_spacing()
# Instructions should be "more complete", but I'll leave it "simple" for this exercise

travel_agent = Agent(name="Travy", 
                   role="Travel Assistant", 
                   instructions="You only plan trips. You only talk about trips. You are here to help the user plan trips.")

query = "Where should I go for vacation in December?"
agent_response = travel_agent.invoke(query)
print_agent_response(travel_agent, agent_response)

print_spacing()

math_tutor_agent = Agent(name="Mathew", 
                   role="Math Assistant", 
                   instructions="You only answer math questions. You only talk math. You are here to help the user with math.")

query = "How do I solve a quadratic equation?"
agent_response = math_tutor_agent.invoke(query)
print_agent_response(math_tutor_agent, agent_response)

print_spacing()

storyteller_agent = Agent(name="Sto", 
                   role="Storyteller Assistant", 
                   instructions="You only tell fictional stories. You only talk about fictional stories. You are here to entertain the user interesting fictional stories. Limit the stories to 500 words.",
                   temperature=1.5)

query = "Tell me a story about a dragon and a wizard"
agent_response = storyteller_agent.invoke(query)
print_agent_response(storyteller_agent, agent_response)

# Just testing
# test_agent = Agent(name="Tom", role="Travel Assistant", instructions="Tell me the distance between city A and city B", model="gpt-4o-mini")
# print("Let's test this workflow:")
# query = input()
# while(query != "0"):
#     print(test_agent.invoke(query))
#     print("What's next?")
#     query = input()