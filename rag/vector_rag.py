from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import faiss
from langchain_classic.chains import RetrievalQA
import os
import pickle
import numpy as np

from dotenv import load_dotenv
load_dotenv()

INDEX_FILE = "vectorstore.faiss"
STORE_FILE = "vectorstore.pkl"


def load_vector_store():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(STORE_FILE):
        raise RuntimeError("Vector store not found. Run ingest.py first.")

    index = faiss.read_index(INDEX_FILE)
    texts = pickle.load(open(STORE_FILE, "rb"))

    emb = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBED_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    return index, texts, emb


def vector_rag_answer(question, index, texts, emb, k=4):
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model= os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0
    )

    # ✅ Embed query
    q_vec = emb.embed_query(question)
    q_vec = np.array([q_vec]).astype("float32")

    # ✅ Search FAISS
    distances, indices = index.search(q_vec, k)

    retrieved_chunks = [texts[i] for i in indices[0] if i < len(texts)]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Use ONLY the context below to answer the question.
    If the answer is not present, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt).content
