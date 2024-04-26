from openai import OpenAI

from councilofelders.agent import Agent
from councilofelders.utils import merge_items_by_role, update_role

GPT4_TURBO = "gpt-4-turbo"
GPT4 = "gpt-4"
GPT4_32K = "gpt-4-32k"
GPT_3_5_TURBO = "gpt-3.5-turbo"

class OpenAIAgent(Agent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(OpenAI(api_key= api_key),
                         model,
                         temperature,
                         name)
        self.add_message_to_history(system_prompt, 'system')

    def add_message_to_history(self, msg, who):
        if (who != 'user') or (who != 'system'):
            who = 'assistant'
        self.history.append({'content': msg, 'role': who})

    def generate_next_message(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=merge_items_by_role(update_role(self.history, self.name)))
        return completion.choices[0].message.content
