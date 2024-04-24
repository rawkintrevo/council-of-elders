from councilofelders.agent import Agent
from councilofelders.utils import merge_items_by_role, update_role

from anthropic import Anthropic


class AnthropicAgent(Agent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(Anthropic(api_key= api_key),
                         model,
                         temperature,
                         name)
        self.system_prompt = system_prompt

    def add_message_to_history(self, msg, who):
        print(f"AnthropicAgent.add_message_to_history called with signature: msg={msg} , who={who}")
        if (who == 'system'):
            print("Anthropic changing role 'system' to 'user'")
            who = 'user'
        if (who != 'user'):
            print(f"Anthropic changing role '{who}' to 'assistant'")
            who = 'assistant'
        print(f"Anthropic appending hx {{role: '{who}', content: '{msg}'}}")

        self.history.append({'content': msg, 'role': who})
        if len(self.history) == 1:
            print(f"AnthropicAgent.add_message_to_history corner case detected")
            self.history[0]['role'] = 'user'
            print(f"AnthropicAgent.add_message_to_history self.history[0]['role']: {self.history[0]['role']}")

    def generate_next_message(self):
        print("AnthropicAgent.generate_next_message...")
        message = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1000, #required
            system=self.system_prompt,
            messages=merge_items_by_role(update_role(self.history, self.name))
        )
        if 'content' in message:
            print(f"AnthropicAgent.generate_next_message message: {message.content[0].text}")
        else:
            print(f"AnthropicAgent.generate_next_message failed message: {message}")
        return message.content[0].text
