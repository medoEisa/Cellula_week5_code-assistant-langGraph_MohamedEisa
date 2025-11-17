from state_graph import State

class ExplainCodeState(State):
    def __init__(self, agent):
        super().__init__('explain', self.action)
        self.agent = agent

    def action(self, ctx):
        # Retrieve similar code examples
        ctx.retrieved_examples = self.agent.retriever.retrieve(ctx.user_input, top_k=3)

        # Include last assistant response (if any) to preserve context
        last_response = ""
        for msg in reversed(ctx.convo_history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break

        prompt_for_llm = f"Previous assistant code:\n{last_response}\n\nUser wants an explanation: {ctx.user_input}"

        ctx.llm_response = self.agent.llm.call(
            prompt_for_llm,
            retrieved_docs_texts=[e['prompt'] + "\n" + e['canonical_solution'] for e in ctx.retrieved_examples],
            include_retrieved_in_output=False,
            user_id=ctx.metadata.get("user_id", "default_user")
        )

        # Save assistant response to conversation history
        ctx.convo_history.append({"role": "assistant", "content": ctx.llm_response})
