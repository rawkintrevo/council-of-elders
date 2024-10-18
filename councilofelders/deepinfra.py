from councilofelders.openai import OpenAIAgent


class DeepInfraAgent(OpenAIAgent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(model, system_prompt, temperature, name, api_key,
                         base_url="https://api.deepinfra.com/v1/openai")

