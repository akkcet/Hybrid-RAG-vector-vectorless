from pageindex import PageIndexClient
import pageindex.utils as utils
from langchain_openai import ChatOpenAI
from openai import AzureOpenAI
import json

import os
import pageindex.utils as utils
from pageindex import PageIndexClient
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

#llm = ChatOpenAI(temperature=0, api_key=os.getenv("AZURE_OPENAI_API_KEY"))
PAGEINDEX_META_FILE = "knowledgebase/results/pageindex_docs.json"
def build_pageindex(pages):
    """
    Initialize PageIndexClient and ingest document pages
    """
    client = PageIndexClient()

    for page in pages:
        client.add_page(
            text=page.page_content,
            page_number=page.metadata.get("page")
        )

    client.build()  # builds PageIndex tree
    return client

def load_pageindex():
    if not os.path.exists(PAGEINDEX_META_FILE):
        raise RuntimeError("PageIndex metadata not found. Run ingest.py.")

    with open(PAGEINDEX_META_FILE) as f:
        docs = json.load(f)

    return PageIndexClient(), docs



#PAGEINDEX_META_FILE = "pageindex/pageindex_docs.json"

def load_pageindex():
    if not os.path.exists(PAGEINDEX_META_FILE):
        raise RuntimeError("PageIndex metadata not found. Run ingest.py.")

    with open(PAGEINDEX_META_FILE) as f:
        docs = json.load(f)

    return PageIndexClient(api_key=os.getenv("pageindex_api_key")), docs


client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
def compress(nodes,doc_id):
        out = []
        for n in nodes:
            entry = {
                "doc_id": doc_id,
                "node_id": n["node_id"],
                "title":   n["title"],
                "page":    n.get("page_index", "?"),
                "summary": n.get("text", "")
            }
            if n.get("nodes"):
                entry["children"] = compress(n["nodes"],doc_id)
            out.append(entry)
        return out
def find_nodes_by_ids(tree: list, target_ids: list) -> list:
    """Recursively walk the tree and collect nodes matching target_ids."""
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found

def find_nodes_by_doc_ids_ug(compressed_tree, doc_id, node_id=None):
    """
    Find nodes matching doc_id (and optionally node_id) from compressed_tree.

    :param compressed_tree: list of lists containing node dicts
    :param doc_id: document id to search for
    :param node_id: optional node_id to narrow the search
    :return: list of matching node dictionaries
    """
    matches = []

    def traverse(nodes):
        for node in nodes:
            if node.get("doc_id") == doc_id:
                if node_id is None or node.get("node_id") == node_id:
                    matches.append(node)

            # recurse into children if present
            if "children" in node and isinstance(node["children"], list):
                traverse(node["children"])

    for group in compressed_tree:
        traverse(group)

    return matches

def find_nodes_by_doc_ids(tree: list, target_ids: list) -> list:
    """Recursively walk the tree and collect nodes matching target_ids."""
    found = []
    for group in tree:
      for node in group:
        if node.get("doc_id") in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found  
def generate_answer(query: str, nodes: list, model: str = "gpt-4o") -> str:
    """
    Takes retrieved nodes as context and generates a grounded answer.
    Instructs the LLM to cite section titles and page numbers.
    """
    if not nodes:
        return "⚠️ No relevant sections found in the document."
    
    # Build context string from retrieved nodes
    flat_nodes = [n for sublist in nodes for n in sublist]
    context_parts = []
    for node in nodes:
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page', '?')}]\n"
            f"{node.get('summary', 'Content not available.')}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def pageindex_rag_answer(question, pi_client, docs):
    compressed_tree = []
    
    client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

    contexts = []
    # Step 1: Tree Search
    for d in docs:
        doc_id = d["doc_id"]

        if not pi_client.is_retrieval_ready(doc_id):
            continue

        # ✅ Reasoning-based retrieval
        #pages = pi_client.submit_query(doc_id, question)
        tree_result  = pi_client.get_tree(doc_id, node_summary=True)
        pageindex_tree = tree_result.get("result", [])
        #print("pageindex tree:  ",json.dumps(pageindex_tree, indent=2))
        compressed_tree.append(compress(pageindex_tree,doc_id))

        prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
            Your task: identify which node IDs most likely contain the answer to the query.
            Think step-by-step about which sections are relevant.

            Query: {question}

            Document Tree:
            {json.dumps(compressed_tree, indent=2)}

            Reply ONLY in this exact JSON format:
            {{
              "thinking": "<your step-by-step reasoning>",
              "node_list": ["doc_id1:node_id1", "doc_id2:node_id2"]
              
            }}"""
    #print("compressed tree:  ",json.dumps(compressed_tree, indent=2))    
    response = client.chat.completions.create(
        model="gpt4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
          )
    search_result = json.loads(response.choices[0].message.content)
    #print("search result:   ", search_result)
    node_ids = search_result.get("node_list", [])
    #print("node_ids: ",node_ids)
    docs_get = [item.split(":") for item in node_ids]
    #print("docs_get: ",docs_get)
    node = []
    for i,k in docs_get:
      
      node.extend(find_nodes_by_doc_ids_ug(compressed_tree, i,k))
      #print("node:  ",node)
    #tree_result  = pi_client.get_tree(doc_id, node_summary=True)
    # Step 2: Retrieve nodes
    #nodes = find_nodes_by_ids(pageindex_tree, node_ids)
    #print("nodes: ",nodes)
    # Step 3: Generate answer
    answer = generate_answer(question, node)
    

    return answer
