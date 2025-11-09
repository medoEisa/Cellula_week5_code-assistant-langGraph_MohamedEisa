import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

load_dotenv()

# --- System prompt (Task 0) ---
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

#Create Chat model 
chat = ChatOpenAI(
    model="meta-llama/llama-3.3-8b-instruct:free",
    temperature=0.6,
    max_tokens=2056,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Memory setup 
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Build system message 
def build_system_message():
    return SystemMessage(content=SYSTEM_PROMPT)

# Ask model with optional retrieval 
def ask_with_retrieval(user_question, retrieved_docs_texts=None):
    """
    Compose messages with system prompt, memory, retrieved context, and user question,
    then invoke the chat model.
    """
    retrieved_docs_texts = retrieved_docs_texts or []

    # Build retrieval context
    context_block = ""
    if retrieved_docs_texts:
        context_block = "\n\n--- Retrieved context (top results) ---\n"
        for i, doc in enumerate(retrieved_docs_texts[:3], start=1):  # top 3
            snippet = doc[:1000].strip()
            context_block += f"[DOC {i}]\n{snippet}\n\n"

    # Include memory recent conversation
    mem_vars = memory.load_memory_variables({})
    mem_msgs = mem_vars.get("chat_history", [])
    memory_block = ""
    if mem_msgs:
        last_turns = mem_msgs[-6:]  
        memory_block = "\n\n--- Recent conversation (memory) ---\n"
        for m in last_turns:
            who = "User" if m.type == "human" else "Assistant"
            memory_block += f"{who}: {m.content}\n"

    # --- Compose messages 
    messages = [
        build_system_message(),
        HumanMessage(content=f"{memory_block}{context_block}\n\nUser question: {user_question}")
    ]

        # Invoke chat
    response = chat.invoke(messages)
    
    memory.save_context(
        {"input": user_question},
        {"output": response.content}
    )

    return response.content

