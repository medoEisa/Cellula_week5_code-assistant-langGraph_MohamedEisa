from typing import Callable, Dict, Any, Optional
from context import Context
class State:
    def __init__(self, name: str, action: Callable[['Context'], Any] = None):
        self.name = name
        self.action = action
        self.transitions = []

    def add_transition(self, condition_fn: Callable[['Context'], bool], target_state_name: str):
        self.transitions.append((condition_fn, target_state_name))

class StateGraph:
    def __init__(self):
        self.states: Dict[str, State] = {}
        self.start_state: Optional[str] = None

    def add_state(self, state: State, start: bool = False):
        self.states[state.name] = state
        if start or self.start_state is None:
            self.start_state = state.name

    def run(self, context: 'Context'):
        current = self.start_state
        while True:
            state = self.states[current]
            if state.action:
                state.action(context)
            triggered = False
            for cond, target in state.transitions:
                try:
                    if cond(context):
                        current = target
                        triggered = True
                        break
                except Exception:
                    continue
            if not triggered:
                break
