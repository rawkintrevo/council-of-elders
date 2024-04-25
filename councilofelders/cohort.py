
class Cohort:
    def __init__(self, agents: list, history: list, verbose: bool):
        self.agents = agents
        self.history = []
        self.load_history(history)
        self.current_agent = 0
        self.verbose = verbose

    def load_history(self, history: list):
        """In firestore, history will be loaded from a document"""
        self.history = history
        for h_i in range(len(history)):

            for a_i in range(len(self.agents)):
                who = ""
                if history[h_i]['name'] == self.agents[a_i].name:
                    who = "user"
                elif history[h_i]['name'] == "system":
                    who = "system"
                else:
                    who = "assistant"
                self.agents[a_i].add_message_to_history(history[h_i]['response'], who)

    def generate_next_message(self, agent:int = None):
        if agent is not None:
            self.current_agent = agent
        else:
            self.current_agent += 1
            if self.current_agent >= len(self.agents):
                self.current_agent = 0
        r = self.agents[self.current_agent].generate_next_message()
        self.add_response(r)

    def add_response(self, response):
        self.history.append({"name": self.agents[self.current_agent].name,
                             "response": response})
        for i in range(len(self.agents)):
            if i != self.current_agent:
                self.agents[i].add_message_to_history(response, 'assistant')
            else:
                self.agents[i].add_message_to_history(response, 'user')
