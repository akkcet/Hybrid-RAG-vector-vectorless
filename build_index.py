import os
import faiss
import pickle
import asyncio
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import aiofiles
from dotenv import load_dotenv
import pageindex
load_dotenv()
import numpy as np
from pageindex import PageIndexClient
import json

PAGEINDEX_META_FILE = "knowledgebase/results/pageindex_docs.json"
KB_FOLDER = "knowledgebase/"
INDEX_FILE = "vectorstore.faiss"
STORE_FILE = "vectorstore.pkl"
PAGEINDEX_DIR = "pageindex_store"
print(os.getenv("AZURE_OPENAI_EMBED_MODEL"))
# ✅ Embedding model

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def build_pageindex_documents(pdf_dir: str):
    pi_client = PageIndexClient(api_key=os.getenv("pageindex_api_key")) 
    doc_registry = []

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, fname)

        print(f"📤 Submitting document to PageIndex: {fname}")
        response = pi_client.submit_document(pdf_path,verify=False)

        doc_id = response["doc_id"]
        print("✅ Document Submitted:", doc_id)

        doc_registry.append({
            "doc_id": doc_id,
            "file": fname
        })

    # ✅ Persist doc_ids for runtime use
    with open(PAGEINDEX_META_FILE, "w") as f:
        json.dump(doc_registry, f, indent=2)

    print("🌳 PageIndex documents registered successfully")

emb = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBED_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ✅ Extract text from supported formats
async def extract_text(path: str) -> str:
    if path.endswith(".txt"):
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    

    return ""


async def build_index():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    texts = []
    print("📚 Loading documents...")
    
    #build_pageindex_documents(KB_FOLDER)
    for fname in os.listdir(KB_FOLDER):
        path = os.path.join(KB_FOLDER, fname)
        content = await extract_text(path)
        if content.strip():
            chunks = splitter.split_text(content)
            texts.extend(chunks)
            
            


    print(f"✅ Total chunks: {len(texts)}")

    # ✅ Embed in async batches
    print("🔍 Generating embeddings (async)...")
    vectors = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = await emb.aembed_documents(batch)
        vectors.extend(vecs)

    X = np.array(vectors).astype("float32")

    print("✅ Building FAISS index...")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    # Save FAISS index + text store
    faiss.write_index(index, INDEX_FILE)
    pickle.dump(texts, open(STORE_FILE, "wb"))

    print("🎉 Index built and saved!")
    
    # ✅ Build + save PageIndex ONCE
    #print("🌳 Building PageIndex...")
    #pageindex.build()
    #pageindex.save(PAGEINDEX_DIR)
    #if pageindex.is_retrieval_ready(doc_id):
        #tree = pageindex.get_tree(doc_id, node_summary=True)['result']


if __name__ == "__main__":
    asyncio.run(build_index())