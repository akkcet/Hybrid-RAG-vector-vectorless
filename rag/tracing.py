from langsmith import traceable

@traceable(name="ChitChat Answer")
def traced_chitchat(question: str, answer: str):
    return answer


@traceable(name="Vector RAG Answer")
def traced_vector_rag(question: str, answer: str):
    return answer


@traceable(name="PageIndex RAG Answer")
def traced_pageindex_rag(question: str, answer: str):
    return answer


@traceable(name="Hybrid Router Decision")
def traced_hybrid_decision(question: str, decision: str):
    return decision
