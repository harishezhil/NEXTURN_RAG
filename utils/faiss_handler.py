import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


model = SentenceTransformer('all-MiniLM-L6-v2')

# def build_faiss_index(chunks):
#     from langchain.vectorstores import FAISS
#     from langchain.embeddings import HuggingFaceEmbeddings

#     # Handle both  dict and string formats
#     texts = [chunk["content"] if isinstance(chunk, dict) else chunk for chunk in chunks]

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.from_texts(texts, embedding=embeddings)
#     return db, texts


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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


# def get_top_chunks(index, chunk_texts, query, top_k=3):
#     # Use LangChain's similarity search API
#     results = index.similarity_search(query, k=top_k)

#     # `results` will be list of Document objects
#     top_chunks = []
#     for doc in results:
#         top_chunks.append({
#             "content": doc.page_content,
#             "metadata": doc.metadata
#         })

#     return top_chunks

def get_top_chunks(index, chunk_texts, query, top_k=3):
    results = index.similarity_search(query, k=3)  # Fetch more than needed
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
        "filename": doc.metadata.get("filename", "Unknown")
    } for doc in top_docs]

