class Agent:
    def __init__(self, client, model, temperature, name):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.name = name
        self.history = []

    def add_message_to_history(self, msg, who):
        self.history.append({'msg': msg, 'who': who})

    def generate_next_message(self):
        pass
