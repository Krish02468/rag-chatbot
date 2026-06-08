# RAG Q&A Chatbot

A Retrieval-Augmented Generation chatbot built from scratch that answers 
questions grounded strictly in a custom ML knowledge base.

## How It Works
1. Documents → vector embeddings (sentence-transformers)
2. Embeddings stored in ChromaDB (vector database)
3. User question → semantic search → top 3 relevant chunks retrieved
4. Chunks + question → LLaMA 3.1 via Groq API → grounded answer
5. Streamlit chat interface with conversation memory

## Tech Stack
- sentence-transformers (all-MiniLM-L6-v2)
- ChromaDB (persistent vector store)
- LLaMA 3.1 via Groq API
- Streamlit

## Run Locally
pip install -r requirements.txt
python ingest_docs.py
streamlit run app.py