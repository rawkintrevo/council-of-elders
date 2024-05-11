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
            "meta/codellama-70b-instruct"
        ]
        if model not in supported_models:
            raise Warning(f"Model {model} is not supported. Supported models are {supported_models}")

        if model == "codellama-70b-instruct":
            super().__init__(replicate.Client(api_token= api_key),
                             "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf",
                             temperature,
                             name)
        else:
            super().__init__(replicate.Client(api_token= api_key),
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
        if "llama-2" in self.model:
            for item in data:
                if item['role'] == 'user':
                    output += "[INST]" + item['content'] + "[/INST]\n"
                else:
                    output += item['content'] + '\n'
        elif "llama-3" in self.model:
            # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
            output = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + self.system_prompt + "<|eot_id|>"
            for item in data:
                if item['role'] == 'user':
                    output += "<|start_header_id>user<|end_header_id|>\n\n" + item['content']
                else:
                    output += "<|start_header_id>assistant<|end_header_id|>\n\n" + item['content']
                output += "<|start_header_id>assistant<|end_header_id|>\n\n"
        elif "codellama-70b" in self.model:
            # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-code-llama-70b
            output = "<s>Source: system\n\n" + self.system_prompt
            for item in data:
                if item['role'] == 'user':
                    output += "<step> Source: user\n\n" + item['content']
                else:
                    output += "<step> Source: assistant\n" + item['content']
                output += "<step> Source: assistant\nDestination: user"
        # Code-llama and code-llama-70b have different tags too

        # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-code-llama

        return output

    def generate_next_message(self):
        hx_str = self._format_list_of_dicts(merge_items_by_role( update_role( self.history, self.name )))
        input_d =  {
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "prompt": hx_str
        }
        if "llama-3" in self.model:
            input_d["max_new_tokens"] = 2048
        elif "llama-2" in self.model:
            input_d["max_new_tokens"] = 1024
        elif "codellama-70b" in self.model:
            input_d["max_tokens"] = 2048
            del input_d['system_prompt']
        resp = self.client.run(self.model,
                               input = input_d)
        resp = "".join(resp) # list of tokens to string
        return resp

