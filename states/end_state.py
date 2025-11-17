# states/end_state.py
from state_graph import State

class EndState(State):
    def __init__(self, agent):
        super().__init__("end", self.action)
        self.agent = agent

    def action(self, ctx):
        # make sure llm_response exists
        out = ctx.llm_response or "(no assistant response)"
        # persist assistant response to convo_history
        ctx.convo_history.append({"role": "assistant", "content": out})
        print("\n===== Assistant Response =====\n")
        print(out)
        print("\n==============================\n")
