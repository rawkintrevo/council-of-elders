from councilofelders.agent import Agent
from councilofelders.utils import merge_items_by_role, update_role

from anthropic import Anthropic

CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

class AnthropicAgent(Agent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(Anthropic(api_key= api_key),
                         model,
                         temperature,
                         name)
        self.system_prompt = system_prompt

    def add_message_to_history(self, msg, who):
        if (who == 'system'):
            who = 'user'
        if (who != 'user'):
            who = 'assistant'

        self.history.append({'content': msg, 'role': who})
        if len(self.history) == 1:
            self.history[0]['role'] = 'user'

    def generate_next_message(self):
        message = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1000, #required
            system=self.system_prompt,
            messages=merge_items_by_role(update_role(self.history, self.name))
        )
        return message.content[0].text
