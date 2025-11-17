import streamlit as st
import json
import os
from context import Context
from graph_builder import build_graph
from utils.retriever import Retriever
from utils.intent_classifier import IntentClassifier
from utils.humaneval_db import init_chroma, store_embeddings
from utils.llm_client import LLMClient


# --- Constants ---
DATA_PATH = "user_data.json"

# --- Load persistent user data ---
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r") as f:
        all_users = json.load(f)
else:
    all_users = {}

# --- Initialize ChromaDB once ---
if "collection" not in st.session_state:
    st.session_state.collection = init_chroma()
    store_embeddings(st.session_state.collection)
collection = st.session_state.collection
retriever = Retriever(collection)

# --- Agent & State Machine ---
class Agent:
    def __init__(self):
        self.retriever = retriever
        self.intent_classifier = IntentClassifier()
        self.llm = LLMClient()
        self.prompt_manager = None

agent = Agent()
sg = build_graph(agent)

if "login_user" not in st.session_state:
    st.session_state.login_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Cellula Code Assistant", layout="wide")
st.title("💻 Cellula Code Assistant Chat")

# --- LOGIN / SIGNUP ---
if st.session_state.login_user is None:
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", key="login_password", type="password")
        if st.button("Login"):
            user = all_users.get(username)
            if user and user["password"] == password:
                st.session_state.login_user = username
                st.session_state.chat_history = user.get("chat_history", [])
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username", key="signup_username")
        new_pass = st.text_input("Password", key="signup_password", type="password")
        if st.button("Signup"):
            if new_user in all_users:
                st.error("User already exists")
            else:
                all_users[new_user] = {
                    "password": new_pass,
                    "chat_history": [],
                    "context": {"user_input": "", "intent": "", "retrieved_examples": [],
                                "prompt": "", "llm_response": "", "convo_history": [], "metadata": {"user_id": new_user}}
                }
                with open(DATA_PATH, "w") as f:
                    json.dump(all_users, f)
                st.success("Signup successful! Please login.")

# --- CHAT ---
else:
    username = st.session_state.login_user
    user_data = all_users.get(username)
    ctx_data = user_data.get("context", {})
    ctx = Context(**ctx_data)

    chat_container = st.container()

    def render_chat():
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="
                            text-align:right; 
                            background-color:#DCF8C6; 
                            color:black;
                            padding:10px; 
                            border-radius:12px; 
                            margin:5px 0; 
                            max-width:70%; 
                            float:right;
                            clear:both;
                            ">
                            {msg['content']}
                        </div>
                        <div style="clear:both"></div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Assistant left side
                    if "```" in msg["content"]:
                        parts = msg["content"].split("```")
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                if part.strip():
                                    st.markdown(
                                        f"""
                                        <div style="
                                            text-align:left; 
                                            background-color:#F1F0F0; 
                                            color:black;
                                            padding:10px; 
                                            border-radius:12px; 
                                            margin:5px 0; 
                                            max-width:70%;
                                            float:left;
                                            clear:both;
                                            ">
                                            {part.strip()}
                                        </div>
                                        <div style="clear:both"></div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.code(part.strip(), language="python")
                    else:
                        st.markdown(
                            f"""
                            <div style="
                                text-align:left; 
                                background-color:#F1F0F0; 
                                color:black;
                                padding:10px; 
                                border-radius:12px; 
                                margin:5px 0; 
                                max-width:70%;
                                float:left;
                                clear:both;
                                ">
                                {msg['content']}
                            </div>
                            <div style="clear:both"></div>
                            """,
                            unsafe_allow_html=True
                        )


    render_chat()

    # --- Input at bottom ---
    user_input = st.text_input("Type your message here...", key="user_input_field")

    col1, col2 = st.columns([1,1])
    with col1:
        send = st.button("Send")
    with col2:
        clear_memory = st.button("Clear Memory")

    if clear_memory:
        agent.llm.clear_memory()
        ctx.convo_history = []
        st.session_state.chat_history = []
        user_data["chat_history"] = []
        user_data["context"] = ctx.__dict__
        with open(DATA_PATH, "w") as f:
            json.dump(all_users, f)
        st.success("Memory cleared.")
        render_chat()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx.user_input = user_input.strip()
        ctx.convo_history.append({"role": "user", "content": user_input})
        render_chat()

        sg.run(ctx)

        # Append assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": ctx.llm_response})
        ctx.convo_history.append({"role": "assistant", "content": ctx.llm_response})

        # Save context & history to persistent storage
        user_data["chat_history"] = st.session_state.chat_history
        user_data["context"] = ctx.__dict__
        all_users[username] = user_data
        with open(DATA_PATH, "w") as f:
            json.dump(all_users, f)

        render_chat()

    # --- Auto-scroll to bottom using JS ---
    st.markdown(
        """
        <script>
        const chatBox = window.parent.document.querySelectorAll('section.main > div.block-container')[0];
        chatBox.scrollTop = chatBox.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True
    )
