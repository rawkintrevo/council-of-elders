from councilofelders.openai import OpenAIAgent


class DeepseekAgent(OpenAIAgent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(model, system_prompt, temperature, name, api_key,
                         base_url="https://api.deepseek.com")

