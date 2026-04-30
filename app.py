# ----- SSL FIX (MUST BE FIRST) -----
#import certifi_win32  # noqa: F401 (required for Windows cert store)
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
from rag.chitchat import chitchat_answer,is_chitchat, classify_intent

# ----- Standard Imports -----
import streamlit as st
from dotenv import load_dotenv

# ----- Local Imports -----
from rag.vector_rag import (
    load_vector_store,
    vector_rag_answer,
)
from rag.pageindex_rag import (
    load_pageindex,
    pageindex_rag_answer,
)

from rag.tracing import (
    traced_chitchat,
    traced_vector_rag,
    traced_pageindex_rag,
    traced_hybrid_decision,
)

from rag.hybrid_router import hybrid_select

# ----- Env -----
load_dotenv()

# ----- Streamlit Config -----
st.set_page_config(
    page_title="Hybrid RAG Chat",
    layout="wide",
)

st.title("🤖 Hybrid RAG Chat (FAISS + PageIndex)")
st.caption("Vector RAG + Reasoning-based PageIndex RAG")

# ----- Load Indexes ONCE -----
@st.cache_resource
def load_resources():
    """
    Loads all retrieval resources.
    NOTHING is built here.
    """
    index, texts, emb = load_vector_store()
    pi_client, pi_docs = load_pageindex()
    return index, texts, emb, pi_client, pi_docs


try:
    index, texts, emb, pi_client, pi_docs = load_resources()
except Exception as e:
    st.error("❌ Failed to load indexes. Run ingest.py first.")
    st.exception(e)
    st.stop()

# ----- Chat UI -----
question = st.chat_input("Ask a question about the indexed documents")

if question:
    
    try:
        intent = classify_intent(question)
        # ✅ Chit‑chat path
        if intent == "CHITCHAT":
            with st.spinner("Thinking..."):
                answer = chitchat_answer(question)
            traced_chitchat(question, answer)
            st.chat_message("user").markdown(question)
            st.chat_message("assistant").write(answer)
            st.caption("💬 Chit‑chat mode")
            st.stop()

        # Vector RAG
        with st.spinner("Thinking..."):
            vector_answer = vector_rag_answer(
                question=question,
                index=index,
                texts=texts,
                emb=emb,
            )
            traced_vector_rag(question, vector_answer)

            # PageIndex RAG
            pageindex_answer = pageindex_rag_answer(
                question=question,
                pi_client=pi_client,
                docs=pi_docs,
            )
            traced_pageindex_rag(question, pageindex_answer)

            # Hybrid routing
            final_answer, mode = hybrid_select(
                question,
                vector_answer,
                pageindex_answer,
            )
            traced_hybrid_decision(question, mode)
        st.chat_message("user").markdown(question)
        st.chat_message("assistant").write(final_answer)
        st.caption(f"🧠 Retrieval mode used: **{mode}**")

    except Exception as e:
        st.error("❌ Error while answering")
        st.exception(e)
