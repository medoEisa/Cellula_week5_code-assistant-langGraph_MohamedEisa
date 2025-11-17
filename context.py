# context.py
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Context:
    user_input: str = ""
    intent: str = ""
    retrieved_examples: List[Dict[str,str]] = field(default_factory=list)
    prompt: str = ""
    llm_response: str = ""
    convo_history: List[Dict[str,str]] = field(default_factory=list)
    metadata: Dict[str,Any] = field(default_factory=dict)
