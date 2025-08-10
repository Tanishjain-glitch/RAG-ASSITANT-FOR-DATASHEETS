import os
import json
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# ===============================
# 1. ENVIRONMENT SETUP
# ===============================
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("key.env"):
    load_dotenv("key.env")

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env or key.env!")

# ===============================
# 2. APP UI SETUP
# ===============================
st.set_page_config(page_title="STM32 RAG Assistant", layout="wide")
st.title("📥 Upload Datasheet & Ask Embedded Questions")

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

if "session_data" not in st.session_state:
    st.session_state.session_data = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ===============================
# 3. FUNCTIONS
# ===============================
def preprocess_uploaded_pdf(uploaded_file):
    """Save uploaded PDF and split into text chunks"""
    temp_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    chunk_file_path = f"chunks_{uuid.uuid4().hex}.txt"
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.page_content.strip() + "\n\n")

    os.remove(temp_path)  # cleanup PDF
    return chunk_file_path, len(chunks)


def build_vector_index(chunk_file_path):
    """Create FAISS index in memory from chunks"""
    loader = TextLoader(chunk_file_path, encoding='utf-8')
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store, len(documents)


def answer_query(query: str, vector_store):
    """Run query through GPT-OSS-20B with FAISS retriever"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(
        model_name="openai/gpt-oss-20b",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    prompt = (
        "You are an STM32 embedded systems expert. "
        "Respond only with clean, valid C code using CMSIS-style register access. "
        "Do not explain anything.\n\n"
        f"{query}"
    )
    result = qa_chain({"query": prompt})
    return result["result"]

# ===============================
# 4. FILE UPLOAD & INDEX BUILDING
# ===============================
uploaded_file = st.file_uploader("Upload STM32 or PIC PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("🔄 Processing PDF into chunks..."):
        chunk_file_path, total_chunks = preprocess_uploaded_pdf(uploaded_file)
        st.success(f"✅ {total_chunks} chunks generated.")

        # Show preview
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            preview = f.read()[:1000]
        st.text_area("📖 Preview of Chunked Text", preview, height=200)

    with st.spinner("⚙️ Building vector database..."):
        st.session_state.vector_store, doc_count = build_vector_index(chunk_file_path)
        st.success(f"✅ Vector DB ready with {doc_count} documents.")

    st.divider()

# ===============================
# 5. QUERY SECTION
# ===============================
if st.session_state.vector_store:
    query = st.text_input(
        "Ask your embedded systems query:",
        placeholder="e.g. Configure TIM2 for PWM output on PA0"
    )

    if query:
        with st.spinner("🤖 Generating response..."):
            try:
                response = answer_query(query, st.session_state.vector_store)
                st.code(response, language="c")
                st.session_state.session_data.append({"query": query, "response": response})
            except Exception as e:
                st.error(f"❌ Failed to generate answer: {e}")

# ===============================
# 6. SESSION LOG
# ===============================
if st.session_state.session_data:
    st.subheader("📝 Session Log")
    for i, log in enumerate(st.session_state.session_data):
        st.markdown(f"**Q{i+1}:** {log['query']}")
        st.code(log['response'], language="c")

    if st.button("💾 Save Session Log"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_file = os.path.join(SESSION_DIR, f"session_{timestamp}.json")
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.session_data, f, indent=2)
        st.success(f"💾 Session saved as `{session_file}`")
