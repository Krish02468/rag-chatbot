import streamlit as st
from ingest_docs import collection, model as embedding_model
from rag_pipeline import get_answer
import os
if not os.path.exists("chroma_store"):
    import ingest_docs
# ---- PAGE SETUP ----
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Q&A Chatbot")
st.caption("Ask anything based on the documents in the knowledge base")

# ---- SESSION STATE ----
# This remembers chat history within the same session
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask a question...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, context = get_answer(user_input)
        st.markdown(answer)

        # Show source documents in expandable section
        if context:
            with st.expander("📄 Source context used"):
                st.text(context)

    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer
    })