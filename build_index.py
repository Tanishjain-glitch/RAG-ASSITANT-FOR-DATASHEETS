# build_index.py — Create FAISS Index from Chunks
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_vector_index(chunk_file_path):
    # Ensure file exists
    chunk_file_path = os.path.abspath(chunk_file_path)
    if not os.path.isfile(chunk_file_path):
        raise FileNotFoundError(f"File not found: {chunk_file_path}")

    # Load chunks
    loader = TextLoader(chunk_file_path, encoding='utf-8')
    documents = loader.load()

    # Use fast, lightweight embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS index (in memory, no disk writes)
    vector_store = FAISS.from_documents(documents, embeddings)

    # Return the store & document count
    return vector_store, len(documents)
