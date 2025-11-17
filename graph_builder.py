from state_graph import StateGraph
from states.chat_state import ChatState
from states.router_state import RouterState
from states.explain_code_state import ExplainCodeState
from states.generate_code_state import GenerateCodeState
from states.end_state import EndState

def build_graph(agent):
    sg = StateGraph()

    chat = ChatState(agent)
    router = RouterState(agent)
    explain = ExplainCodeState(agent)
    generate = GenerateCodeState(agent)
    end = EndState(agent)

    # Add states
    sg.add_state(chat, start=True)
    sg.add_state(router)
    sg.add_state(explain)
    sg.add_state(generate)
    sg.add_state(end)

    # Define transitions
    chat.add_transition(lambda ctx: True, "router")
    router.add_transition(lambda ctx: ctx.intent == "explain", "explain")
    router.add_transition(lambda ctx: ctx.intent == "generate", "generate")
    router.add_transition(lambda ctx: ctx.intent not in ("explain","generate"), "end")
    explain.add_transition(lambda ctx: True, "end")
    generate.add_transition(lambda ctx: True, "end")

    return sg
