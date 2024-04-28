from councilofelders.agent import Agent

import replicate
class ReplicateLlamaAgent(Agent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        supported_models = [
            "meta/llama-2-7b-chat",
            "meta/llama-2-13b-chat",
            "meta/llama-2-70b-chat",
            "meta/meta-llama-3-8b-instruct",
            "meta/meta-llama-3-70b-instruct",
        ]
        if model not in supported_models:
            raise Warning(f"Model {model} is not supported. Supported models are {supported_models}")

        if model == "meta/meta-llama-3-70b-instruct":
            raise Warning("llama-3-70b-instruct seems to have a glitch where "
                          "the system prompt is not being used. FYI")
        super().__init__(replicate.Client(api_key),
                         model,
                         temperature,
                         name)

    def add_message_to_history(self, msg, who):
        if (who != 'user') or (who != 'system'):
            who = 'assistant'
        self.history.append({'content': msg, 'role': who})

    def generate_next_message(self):
        resp = self.client.run(self.model,
                               input = {
                                   "temperature": self.temperature,
                                   "system_prompt": self.system_prompt,
                                   "prompt": self.history
                               })
        return resp

