from councilofelders.openai import OpenAIAgent


class GemeniAgent(OpenAIAgent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        super().__init__(model, system_prompt, temperature, name, api_key,
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


