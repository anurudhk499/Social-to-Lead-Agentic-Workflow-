"""
rag_pipeline.py
---------------
Builds a local RAG pipeline from autostream_kb.json using:
- HuggingFace sentence-transformers for embeddings (free, local)
- FAISS vector store for similarity search
- Returns top-k relevant chunks for a given query
"""

import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge_base() -> list[Document]:
    """Convert the JSON knowledge base into LangChain Document chunks."""
    with open(KB_PATH, "r") as f:
        kb = json.load(f)

    docs = []

    # --- Company overview ---
    docs.append(Document(
        page_content=f"AutoStream is a SaaS company. Tagline: {kb['tagline']}",
        metadata={"source": "overview"}
    ))

    # --- Pricing plans ---
    for plan in kb["plans"]:
        features_text = ", ".join(plan["features"])
        content = (
            f"{plan['name']}: Costs {plan['price']}. "
            f"Features include: {features_text}."
        )
        docs.append(Document(page_content=content, metadata={"source": "pricing"}))

    # --- Policies ---
    for policy in kb["policies"]:
        docs.append(Document(page_content=f"Policy: {policy}", metadata={"source": "policy"}))

    # --- FAQ ---
    for faq in kb["faq"]:
        content = f"Q: {faq['question']} A: {faq['answer']}"
        docs.append(Document(page_content=content, metadata={"source": "faq"}))

    return docs


def build_vector_store() -> FAISS:
    """Build FAISS vector store from knowledge base documents."""
    docs = load_knowledge_base()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


class RAGRetriever:
    """Singleton RAG retriever – built once, reused across turns."""

    def __init__(self, k: int = 3):
        print("🔧 Building RAG vector store (first run may take ~30s to download embeddings)...")
        self.vectorstore = build_vector_store()
        self.k = k
        print("✅ RAG pipeline ready.\n")

    def retrieve(self, query: str) -> str:
        """Return a single string of the top-k relevant KB chunks."""
        results = self.vectorstore.similarity_search(query, k=self.k)
        context = "\n".join([f"- {doc.page_content}" for doc in results])
        return context
