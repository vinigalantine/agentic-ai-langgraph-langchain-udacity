"""
    Exercise 02 - Create Agent from scra
    
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
    def __init__(self, name:str="Franky", role:str="General Assistant", instructions:str="Help the user with general queries.", model:str = "gpt-4o-mini", temperature:float=1.0) -> None:
        self._client = OpenAI() # Would be good to be "client-agnostic" but for pratical reasons let's keep it OpenAI
        
        try:
            disponible_models = [openai_model.id for openai_model in self._client.models.list()]
        except Exception:
            disponible_models = ["gpt-4o-mini", "gpt-4o"]
            
        if model not in disponible_models:
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


print("Let's test this workflow:")
test_agent = Agent(name="Tom", role="Travel Assistant", instructions="Tell me the distance between city A and city B", model="gpt-4o-mini")
query = input()
while(query != "0"):
    print(test_agent.invoke(query))
    print("What's next?")
    query = input()