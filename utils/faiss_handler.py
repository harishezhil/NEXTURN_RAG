"""
WHAT IS THIS FILE??
----------

FAISS-based embedding index builder and retriever — lets your LLM find the good stuff faster.

Functions:
----------

- build_faiss_index(chunks):
    Turns your document chunks into a searchable FAISS index. Think: Ctrl+F, but smarter.
    Uses HuggingFace's MiniLM embeddings and wraps them with LangChain's FAISS.

- get_top_chunks(index, chunk_texts, query, top_k=3):
    Retrieves the top-k most relevant chunks for a query.
    If a year like "2024" is mentioned in the query, it prioritizes results mentioning that year.
"""

import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["content"],
                metadata={"filename": chunk.get("filename", "Unknown")}
            )
        )

    db = FAISS.from_documents(documents, embedding=embeddings)
    return db, documents



def get_top_chunks(index, chunk_texts, query, top_k=3):
    results = index.similarity_search(query, k=3) 
    year_match = re.search(r"\b(20\d{2})\b", query)

    if year_match:
        year = year_match.group(1)
        filtered = [
            doc for doc in results if year in doc.page_content
        ]
        # Fallback to original if not enough
        top_docs = filtered[:top_k] if len(filtered) >= top_k else results[:top_k]
    else:
        top_docs = results[:top_k]

    return [{
        "content": doc.page_content,
        "metadata": doc.metadata
    } for doc in top_docs]

