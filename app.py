import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- SETUP (runs once) ----
@st.cache_resource
def setup():
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="/tmp/chroma_store")
    collection = chroma_client.get_or_create_collection("my_docs")
    
    # Auto-ingest documents if collection is empty
    if collection.count() == 0:
        docs = []
        for filename in os.listdir("documents"):
            if filename.endswith(".txt"):
                with open(f"documents/{filename}", "r") as f:
                    text = f.read()
                    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                    docs.extend(chunks)
        embeddings = embedding_model.encode(docs).tolist()
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
    
    # Connect to Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    return embedding_model, collection, groq_client

embedding_model, collection, groq_client = setup()

# ---- RAG FUNCTION ----
def get_answer(user_question):
    # Embed the question
    question_embedding = embedding_model.encode([user_question]).tolist()
    
    # Retrieve top 3 relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    context = "\n\n".join(results['documents'][0])
    
    # Build prompt
    prompt = f"""You are a helpful assistant.
Answer the question based only on the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {user_question}

Answer:"""
    
    # Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    return answer, context

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Q&A Chatbot")
st.caption("Ask anything based on the documents in the knowledge base")

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask a question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, context = get_answer(user_input)
        st.markdown(answer)

        if context:
            with st.expander("📄 Source context used"):
                st.text(context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })