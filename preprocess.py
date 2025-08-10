from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uuid

CHUNKS_DIR = "chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

def preprocess_uploaded_pdf(uploaded_file):
    # Save uploaded PDF temporarily
    temp_path = os.path.join(CHUNKS_DIR, f"temp_{uuid.uuid4().hex}.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF into pages
    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    # Chunk text for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Save chunks to text file
    chunk_file_path = os.path.join(CHUNKS_DIR, "chunks.txt")
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            content = chunk.page_content.strip()
            if content:  # Skip empty chunks
                f.write(content + "\n\n")

    # Optional: delete temp PDF after processing
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return chunk_file_path, len(chunks)
