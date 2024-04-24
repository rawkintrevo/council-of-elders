from councilofelders.agent import Agent
from councilofelders.utils import merge_items_by_role, update_role

import google.generativeai as genai


class GemeniAgent(Agent):
    def __init__(self, model, system_prompt, temperature, name, api_key):
        genai.configure(api_key=api_key)
        self.history = []
        client = genai.GenerativeModel(model,
                                       generation_config=genai.GenerationConfig(temperature=temperature),
                                       system_instruction=[system_prompt])
        self.chat = client.start_chat(history=[])
        self.current_prompt = ""
        super().__init__(client,
                         model,
                         temperature,
                         name)
        self.add_message_to_history(system_prompt, 'system')

    def add_message_to_history(self, msg, who):

        # who = 'user' | 'model'
        if who != 'user':
            who = 'model'
        print(f"Gemeni appending hx {{role: '{who}', content: '{msg}'}}")
        self.chat.history.append({'parts': [msg], 'role': who})
        self.history = self.chat.history

    def generate_next_message(self):
        msg = self.chat.history.pop()
        self.current_prompt = ' '.join(msg['parts'])
        self.chat.history = merge_items_by_role(update_role(self.chat.history, self.name))
        response = self.chat.send_message(self.current_prompt, safety_settings={'HARASSMENT':'block_none'})
        return(response.candidates[0].content.parts[0].text)
