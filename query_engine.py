# query_engine.py — Using GPT-OSS-20B via OpenRouter (In-Memory Mode)
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Load API key from .env or key.env
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("key.env"):
    load_dotenv("key.env")

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("❌ OPENROUTER_API_KEY not found! Please check your .env or key.env file.")

# Function to create a QA chain from an existing vector store
def create_qa_chain(vector_store):
    llm = ChatOpenAI(
        model_name="openai/gpt-oss-20b",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# Main function for answering queries
def answer_query(query: str, vector_store) -> str:
    """
    Query STM32 RAG with GPT-OSS-20B, return C code only.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

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

