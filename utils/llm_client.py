import os
from dotenv import load_dotenv
load_dotenv()

import json
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are an AI tutor specialized in Artificial Intelligence and Machine Learning.

**Role and Expertise:**
- You are an educational assistant focused exclusively on AI/ML topics
- Your domain includes: Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, 
  Natural Language Processing, Computer Vision, Data Science, and related programming concepts
- You provide clear, logical explanations as if teaching a student

**Response Guidelines:**
- Use examples related to Python, AI algorithms, ML models, and data processing
- Keep explanations concise, educational, and practical
- Maintain a formal but friendly tone
- Encourage learning within the AI domain
- Use technical terms when appropriate but explain complex concepts accessibly

**Strict Knowledge Boundaries:**
You must decline to answer any question that is unrelated to AI, data science, or programming.

**Forbidden Topics:**
- emotional, political, religious, or entertainment questions
- General knowledge, history, or current events outside AI
- Any unethical, private, or unsafe requests
- Medical, legal, or financial advice
- Questions about other AI assistants or their capabilities

**When asked about unrelated topics:**
Politely respond with:
"I'm sorry, but I can only discuss topics related to Artificial Intelligence, Machine Learning, NLP, Data Science, and related technical fields. How can I help you with AI-related topics?"

**Response Structure:**
1. Provide a clear, direct answer to the AI-related question
2. Include relevant examples or analogies when helpful
3. Reference technical concepts accurately
4. Suggest related topics for deeper exploration
5. Maintain educational value throughout

Stay consistent, professional, and always focus on advancing the user's understanding of AI technologies."""

class LLMClient:
    def __init__(self):
        self.memory = MemorySaver()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_key:
            print(" OPENROUTER_API_KEY not found in environment. LLM calls may fail.")

    def clear_memory(self):
        """Clear conversation memory safely."""
        if self.memory:
            try:
                # MemorySaver internal storage
                if hasattr(self.memory, "_memory") and isinstance(self.memory._memory, dict):
                    self.memory._memory.clear()
                # Some versions have 'store'
                elif hasattr(self.memory, "store") and hasattr(self.memory.store, "clear"):
                    self.memory.store.clear()
                print(" Conversation memory cleared.")
            except Exception as e:
                print(f" Failed to clear memory: {e}")
        else:
            print(" No memory object available to clear.")

        

    def build_system_message(self, overrides: str = None):
        return overrides or SYSTEM_PROMPT

    def _call_openrouter_http(self, prompt: str):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6,
            "max_tokens": 1500
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    def call(self, user_input: str, retrieved_docs_texts=None, include_retrieved_in_output=True, system_overrides: str = None, user_id="default_user"):
        retrieved_docs_texts = retrieved_docs_texts or []

        # Retrieved 
        context_block = ""
        if retrieved_docs_texts and include_retrieved_in_output:
            context_block = "\n\n--- Retrieved examples ---\n"
            for i, doc in enumerate(retrieved_docs_texts[:3], start=1):
                snippet = doc[:1200].strip()
                context_block += f"[Example {i}]\n{snippet}\n\n"

        # Load previous conversation memory
        memory_block = ""
        if self.memory:
            try:
                mem_state = self.memory.load_memory_variables({user_id: {}})
                mem_msgs = mem_state.get("chat_history", [])
                if mem_msgs:
                    last_turns = mem_msgs[-6:]
                    memory_block = "\n\n--- Recent conversation ---\n"
                    for m in last_turns:
                        memory_block += f"{m.get('role','User')}: {m.get('content','')}\n"
            except Exception:
                memory_block = ""

        # Final prompt
        system_msg = self.build_system_message(system_overrides)
        final_prompt = f"{system_msg}\n\n{memory_block}\n{context_block}\nUser question: {user_input}"

        # Call OpenRouter
        try:
            content = self._call_openrouter_http(final_prompt)
            # Save to memory
            if self.memory:
                try:
                    self.memory.save_context({user_id: {"input": user_input}}, {user_id: {"output": content}})
                except Exception:
                    pass
            return content
        except Exception as e:
            return f"(LLM call failed) {str(e)}"
