import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Use persistent client — saves to disk, doesn't reset
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("my_docs")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---- MAIN FUNCTION ----
def get_answer(user_question):
    
    # STEP 1: Convert user question to embedding (numbers)
    # Same process as we did for documents
    question_embedding = embedding_model.encode([user_question]).tolist()

    # STEP 2: Search ChromaDB for top 3 most relevant chunks
    # ChromaDB compares question embedding vs document embeddings
    # Returns the closest matches
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    
    # Join the 3 chunks into one block of context
    context = "\n\n".join(results['documents'][0])

    # STEP 3: Send context + question to Groq LLM
    # We build a prompt that tells the LLM:
    # "Here are some documents. Answer the question using only these."
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

    # Extract and return the answer text
    answer = response.choices[0].message.content
    return answer, context  # return context too so we can show sources