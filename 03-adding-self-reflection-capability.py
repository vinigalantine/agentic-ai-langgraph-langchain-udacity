"""
    Exercise 03 - Adding Self-Reflection Capability
    
    Challenge: You are tasked with upgrading the existing agent. This version can learn from its previous answers, identify mistakes, and refine its responses automatically.
    
    Edit: Forgot to put this note here, but the exercise asked for a "Memory" class and some different approches, I'm cooler thant that
    and opted to do it a bit different. 😎 (just opted to keep it simple 😂)
"""

from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Copied Agent class from exercise 2, I could use import but for "changing the class" purposes I will just copy it whenever needed.
# Could also leave to "check differences" with the git versioning but this is just an exercise bruh
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
    
    def invoke(self, message: str, self_reflect: bool = False, max_iter: int = 1) -> str:
        if self_reflect and not 1 <= max_iter <= 3:
            raise ValueError(f"The max_iter value {max_iter} is invalid, please provide a number more or equal to 1 and less or equal to 3.")
        
        message_chat = {"role": "user", "content": message }
        
        try:    
            response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=self._messages + [message_chat]
            )
        except Exception as e:
            raise RuntimeError("Failed to get the agent response") from e
        
        if self_reflect:
            for i in range(max_iter):
                system_critique_message = {"role": "system", "content": "Reflect on your previous response. Please analyse, think about, identify errors, clarify, improve clarity and refine your last message." }
                last_message = {"role": "assistant", "content": response.choices[0].message.content or "" } 
                
                # logging
                logger.info(f'Iteration: {i+1} - Last message: "{last_message["content"]}"')
                
                try:    
                    response = self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=self._messages + [message_chat, last_message, system_critique_message]
                    )
                except Exception as e:
                    raise RuntimeError("Failed to get the agent response") from e
        
        self._messages.append(message_chat)
        
        agent_response = response.choices[0].message.content
        
        if agent_response:
            self._messages.append({"role": "assistant", "content": agent_response })
        
        return agent_response or "[Agent did not respond]"
    
default_agent = Agent()
query = "What is the capital of France?"
agent_response = default_agent.invoke(message=query, self_reflect=True, max_iter=2)

print("\n" + "="*100 + "\n")
print("-"*50 + " FINAL ANSWER " + "-"*50)
print("\n" + "="*100 + "\n")
print(f"{default_agent.role} - {default_agent.name} the agent: {agent_response}")