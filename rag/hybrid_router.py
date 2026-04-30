from langchain_openai import ChatOpenAI
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

def hybrid_select(question, vector_ans, pageindex_ans):
    judge_prompt = f"""
    Question: {question}

    Answer A (Vector / FAISS):
    {vector_ans}

    Answer B (PageIndex):
    {pageindex_ans}

    Which answer is more accurate and grounded?
    Return A or B with a brief reason.
    """

    decision =  llm.chat.completions.create(
        model="gpt4o",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    if "A" in decision:
        return vector_ans, "VECTOR_RAG"
    else:
        return decision.choices[0].message.content.strip(), "PAGEINDEX_RAG"
