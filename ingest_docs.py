 
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("my_docs")

# Read all .txt files from documents folder
documents = []
for filename in os.listdir("documents"):
    if filename.endswith(".txt"):
        with open(f"documents/{filename}", "r") as f:
            text = f.read()
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
            documents.extend(chunks)

print(f"Loaded {len(documents)} chunks")

# Create embeddings and store
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("✅ Documents stored in ChromaDB successfully!")
if __name__ == "__main__":
    print("Ingestion complete")