"""Streamlit UI for the HR policy assistant. Run with: streamlit run app.py"""

import streamlit as st

from hr_assistant import config
from hr_assistant.pipeline import ask_agent, build_hr_assistant
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

SAMPLE_QUESTIONS = [
    "What are the key steps in the employee hiring process?",
    "What documents are generally required when joining a new company?",
    "How can an employee apply for leave?",
    "What should I do if my salary credited is incorrect?",
    "How should I report workplace harassment or misconduct?",
]


@st.cache_resource(show_spinner="Loading HR knowledge base and assistant...")
def load_assistant():
    return build_hr_assistant()


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def ask(question: str):
    agent = load_assistant()
    with st.chat_message("assistant"):
        with st.spinner("Looking up HR guidance..."):
            answer = ask_agent(agent, question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="🧑‍💼",
        layout="wide",
    )
    init_session()

    with st.sidebar:
        st.title("HR Assistant")
        st.caption("Ask questions about hiring, onboarding, leave, payroll, and workplace conduct.")
        st.divider()
        st.markdown("**Knowledge base**")
        st.code(config.file_path, language=None)
        st.markdown("**Model**")
        st.write(config.LLM_model)
        st.markdown("**Embeddings**")
        st.write(config.Embeddings_model)
        st.divider()
        st.markdown("**Try a question**")
        for i, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, use_container_width=True, key=f"sample_{i}"):
                st.session_state.pending_question = question
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()
        st.caption("Answers come from the HR knowledge base. Company-specific or sensitive cases should still go to HR.")

    st.header("Chat with the HR assistant")
    st.write("Use a starter question in the sidebar or type your own. The assistant searches the HR document before answering.")

    typed = st.chat_input("Ask about leave, hiring, payroll, conduct...")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if not st.session_state.messages and not question:
        st.info("Conversation is empty. Ask about hiring, leave, payroll, onboarding, or workplace conduct.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        ask(question)


if __name__ == "__main__":
    main()
