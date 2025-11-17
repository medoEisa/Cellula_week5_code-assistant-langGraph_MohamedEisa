from state_graph import State

class RouterState(State):
    def __init__(self, agent):
        super().__init__("router", self.action)
        self.agent = agent

    def action(self, ctx):
        ctx.intent = self.agent.intent_classifier.infer(ctx.user_input)
        print(f"[RouterState] Inferred intent: {ctx.intent}")
