import streamlit as st

# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar settings. Returns a dict of user-configured parameters."""
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve per query.",
        )

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Your question is embedded\n"
            "2. Relevant document chunks are retrieved\n"
            "3. An LLM generates an answer based on the context"
        )

    return {"top_k": top_k}


def render_message(message: dict) -> None:
    """Render a single chat message with optional source expander."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for message in st.session_state.messages:
        render_message(message)


def get_bot_response(query: str, top_k: int) -> tuple[str, list[str]]:
    from rag_pipeline import ask, RAGConfig

    answer, sources = ask(query, config=RAGConfig(top_k=top_k, min_relevance=0.6))
    return answer, sources


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Customer Service Chatbot",
        page_icon="🚗",
        layout="centered",
    )

    st.title("🚗 Customer Service Chatbot")
    st.caption("Ask questions about vehicles, services, warranty, and more.")

    # Sidebar
    settings = render_sidebar()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        answer, sources = get_bot_response(prompt, top_k=settings["top_k"])

        response = {"role": "assistant", "content": answer, "sources": sources}
        render_message(response)
        st.session_state.messages.append(response)


if __name__ == "__main__":
    main()
