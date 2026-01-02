from openai import OpenAI
import tiktoken
import json
from datetime import datetime
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096

class ConversationManager(object):
    # Настраивает среду чат-бота
    def __init__(self):
        # do not create Streamlit widgets during construction
        # title and sidebar widgets will be rendered explicitly by calling render_settings()
        self.conversation_history = []
        # set defaults for persona/system_message so instance is in a valid state
        self.persona = st.session_state.get("persona", "Helpful Assistant")
        self.system_message = st.session_state.get("system_message", "You are a helpful AI assistant that provides clear answers.")

    def render_settings(self):
        # render UI including title (moved here so it's called during normal script execution)
        st.title("AI ChatBot")
        sidebar = st.sidebar
        sidebar.header("Settings")

        # # OpenAI API key input
        # sidebar.text_input(
        #     "OpenAI API Key",
        #     value=st.session_state.get("api_key", DEFAULT_API_KEY) or "",
        #     type="password",
        #     key="api_key",
        #     help="Paste your OpenAI API key here."
        # )

        # Model selection
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        sidebar.selectbox(
            "OpenAI Model",
            options=model_options,
            index=model_options.index(st.session_state.get("model", DEFAULT_MODEL)) if st.session_state.get("model", DEFAULT_MODEL) in model_options else 0,
            key="model",
        )

        # temperature slider
        sidebar.slider(
            "Chat Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("temperature", DEFAULT_TEMPERATURE)),
            step=0.01,
            key="temperature",
        )
        # token budget slider
        sidebar.slider(
            "Token Budget",
            min_value=256,
            max_value=4096,
            value=int(st.session_state.get("token_budget", DEFAULT_TOKEN_BUDGET)),
            step=256,
            key="token_budget",
        )

        # persona selectbox
        persona_options = [
            "Helpful Assistant",
            "Formal Mentor",
            "Funny Companion",
            "Concise Expert",
            "Storyteller",
            "Custom",
        ]
        default_persona = st.session_state.get("persona", "Helpful Assistant")
        sidebar.selectbox(
            "Chatbot Personality",
            options=persona_options,
            index=persona_options.index(default_persona) if default_persona in persona_options else 0,
            key="persona",
        )

        # read values from session_state (Streamlit keeps these up to date)
        self.temperature = st.session_state.get("temperature", DEFAULT_TEMPERATURE)
        self.token_budget = st.session_state.get("token_budget", DEFAULT_TOKEN_BUDGET)
        self.persona_selectbox = st.session_state.get("persona", default_persona)

        # If user selected Custom, show textbox and an explicit button to apply the custom message
        if self.persona_selectbox == "Custom":
            custom_text = sidebar.text_area(
                "Custom system message",
                value=st.session_state.get("system_message", ""),
                key="custom_system_textbox",
                help="Enter the system message that will define the chatbot's persona.",
                height=120,
            )
            if sidebar.button("Apply Custom Persona", key="apply_custom_persona_btn"):
                self.set_custom_system_message(custom_text)
        else:
            # non-custom personas: set persona immediately (keeps system_message consistent)
            self.set_persona(self.persona_selectbox)

        # Reset conversation history button (placed last so Streamlit processes other inputs first)
        if sidebar.button("Reset Conversation History", key="reset_history_btn"):
            # set a flag so the actual reset and rerun happen in the main execution flow
            st.session_state["reset_requested"] = True

    # Настраивает личность чат-бота
    def set_persona(self, persona_name=None):
        persona = persona_name or st.session_state.get("persona", "Helpful Assistant")
        if persona == "Helpful Assistant":
            system_msg = "You are a helpful, polite assistant that provides clear and friendly answers."
        elif persona == "Formal Mentor":
            system_msg = "You are a formal mentor: concise, respectful, and educational in tone."
        elif persona == "Funny Companion":
            system_msg = "You are a witty and playful companion, using light humor where appropriate."
        elif persona == "Concise Expert":
            system_msg = "You are an expert who gives short, precise, technical answers with minimal fluff."
        elif persona == "Storyteller":
            system_msg = "You are an imaginative storyteller who responds with vivid narrative and detail."
        else:
            system_msg = "You are a helpful AI assistant that provides clear answers."
        # store persona and system message on instance and session state for use elsewhere
        self.persona = persona
        self.system_message = system_msg
        # Do NOT overwrite the widget-managed "persona" key here; only update system_message
        st.session_state["system_message"] = system_msg

    # Позволяет пользователям устанавливать собственное системное сообщение
    def set_custom_system_message(self, custom_msg: str):
        # validate and apply custom message; if empty, warn and do not set
        if not custom_msg or not custom_msg.strip():
            st.warning("Please enter a custom system message before applying.")
            return
        system_msg = custom_msg.strip()
        self.persona = "Custom"
        self.system_message = system_msg
        st.session_state["persona"] = "Custom"
        st.session_state["system_message"] = system_msg

    # Генерирует ответы на основе ввода пользователя
    def chat_completion(self, user_input: str, token_budget=None, temperature=None, persona=None):
        if not user_input or not user_input.strip():
            return
        token_budget = token_budget if token_budget is not None else st.session_state.get("token_budget", DEFAULT_TOKEN_BUDGET)
        temperature = temperature if temperature is not None else st.session_state.get("temperature", DEFAULT_TEMPERATURE)
        persona = persona if persona is not None else st.session_state.get("persona", getattr(self, "persona", "Helpful Assistant"))
        system_msg = getattr(self, "system_message", st.session_state.get("system_message", ""))

        # Prepare messages for OpenAI (exclude the just-submitted user message)
        messages = [{"role": "system", "content": system_msg}]
        for msg in self.conversation_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        # Add the new user message only for the API call, not to the history yet
        messages.append({"role": "user", "content": user_input})

        api_key = st.session_state.get("api_key", DEFAULT_API_KEY)
        model = st.session_state.get("model", DEFAULT_MODEL)

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=token_budget,
            )
            assistant_text = response.choices[0].message.content.strip()
        except Exception as e:
            assistant_text = f"[Error contacting OpenAI API: {e}]"

        # Now append the user and assistant messages to the conversation history
        user_msg = {"role": "user", "content": user_input, "timestamp": datetime.utcnow().isoformat()}
        self.conversation_history.append(user_msg)
        assistant_msg = {"role": "assistant", "content": assistant_text, "timestamp": datetime.utcnow().isoformat()}
        self.conversation_history.append(assistant_msg)
        st.session_state["conversation_history"] = self.conversation_history

    # New: render full conversation history in main area
    def display_conversation_history(self):
        history = st.session_state.get("conversation_history", self.conversation_history)
        if not history:
            st.info("No messages yet. Start the conversation below.")
            return
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                # include persona tag if available in stored content
                st.markdown(f"**Assistant:** {content}")
            else:
                st.markdown(f"**{role.capitalize()}:** {content}")

    # Очищает историю чата
    def reset_conversation_history(self):
        # clear in-memory and session-state history
        self.conversation_history = []
        st.session_state["conversation_history"] = []
        # optional: clear any stored messages related to model/state
        st.session_state.pop("system_message", None)
        st.session_state.pop("persona", None)
        # reset_requested will be cleared by the caller after processing

if __name__ == "__main__":
    # Initialize session state only once to prevent reinitialization
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.conversation_history = []
        st.session_state.api_key = DEFAULT_API_KEY or ""
        st.session_state.model = DEFAULT_MODEL
        st.session_state.temperature = DEFAULT_TEMPERATURE
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
        st.session_state.token_budget = DEFAULT_TOKEN_BUDGET
        # default persona
        st.session_state.persona = st.session_state.get("persona", "Helpful Assistant")
        # create conversation_manager but do not rely on widgets in __init__
        st.session_state["conversation_manager"] = ConversationManager()

    # Ensure conversation_manager exists (fallback if missing due to earlier rerun)
    if "conversation_manager" not in st.session_state:
        st.session_state["conversation_manager"] = ConversationManager()

    # Assign the initialized session-state variable to a dedicated local variable
    x = st.session_state["conversation_manager"]

    # Now render settings (widgets) during normal Streamlit execution flow
    x.render_settings()

    # If a reset was requested from the sidebar, perform it now (do NOT call experimental_rerun here)
    if st.session_state.pop("reset_requested", False):
        x.reset_conversation_history()

    # Only update conversation history if a new user input is received
    user_input = st.chat_input("Type a message...")

    if user_input:
        x.chat_completion(
            user_input,
            token_budget=st.session_state.get("token_budget"),
            temperature=st.session_state.get("temperature"),
            persona=st.session_state.get("persona"),
        )
    # Always display the current conversation history (once per rerun)
    x.display_conversation_history()

