from context import Context
from graph_builder import build_graph
from utils.retriever import Retriever
from utils.intent_classifier import IntentClassifier
from utils.humaneval_db import init_chroma, store_embeddings
from utils.llm_client import LLMClient

collection = init_chroma()
store_embeddings(collection)

retriever = Retriever(collection=collection)

class Agent:
    def __init__(self):
        self.context = Context()
        self.retriever = retriever
        self.intent_classifier = IntentClassifier()
        self.llm = LLMClient()
        self.prompt_manager = None  

agent = Agent()

sg = build_graph(agent)

print("LangGraph Humaneval AI Tutor (type 'exit' to quit, 'clear' to reset memory)")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if user_input.lower() == "clear":
        agent.llm.clear_memory()
        agent.context = Context()  # reset conversation context
        continue

    # update the agent's shared context object and run the state graph
    agent.context.user_input = user_input
    # ensure convo_history is preserved inside agent.context
    agent.context.convo_history.append({"role": "user", "content": user_input})
    sg.run(agent.context)
