from councilofelders.agent import Agent

import replicate
from councilofelders.utils import merge_items_by_role, update_role


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

        # if model == "meta/meta-llama-3-70b-instruct":
        #     raise Warning("llama-3-70b-instruct seems to have a glitch where "
        #                   "the system prompt is not being used. FYI")
        super().__init__(replicate.Client(api_key),
                         model,
                         temperature,
                         name)
        self.system_prompt = system_prompt

    def add_message_to_history(self, msg, who):
        if (who != 'user') or (who != 'system'):
            who = 'assistant'
        self.history.append({'content': msg, 'role': who})

    def _format_list_of_dicts(self, data):
        output = ""
        for item in data:
            if item['role'] == 'user':
                output += "[INST]" + item['content'] + "[/INST]\n"
            else:
                output += item['content'] + '\n'
        return output

    def generate_next_message(self):
        hx_str = self._format_list_of_dicts(merge_items_by_role( update_role( self.history, self.name )))
        resp = self.client.run(self.model,
                               input = {
                                   "temperature": self.temperature,
                                   "system_prompt": self.system_prompt,
                                   "prompt": hx_str
                               })
        return resp

