from state_graph import State

class ChatState(State):
    def __init__(self, agent):
        super().__init__("chat", self.action)
        self.agent = agent

    def action(self, ctx):
        # store user input in context convo_history
        ctx.convo_history.append({"role": "user", "content": ctx.user_input})
        #  add to LLM memory if available (so memory-aware LLM responses later)
        try:
            if hasattr(self.agent.llm, "memory"):
                self.agent.llm.memory.save_context({"input": ctx.user_input}, {})
        except Exception:
            pass
        print(f"[ChatState] stored user input in memory (messages={len(ctx.convo_history)})")
