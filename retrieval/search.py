"""
Vector Search Module

Performs similarity search against a FAISS index using query embeddings.
The top_k parameter defaults to config.yaml → retrieval.vector_top_k.
"""

import numpy as np
from embeddings.embedder import embed_texts

# [Step 1] Import config loader for vector_top_k default
from config.settings import get as cfg


def search(query, index, documents, top_k=None):
    """
    Search for the most relevant document chunks using vector similarity.

    Args:
        query: The user's question string
        index: FAISS index containing document embeddings
        documents: List of text chunks (parallel to index vectors)
        top_k: Number of results to return (defaults to config.yaml → retrieval.vector_top_k)

    Returns:
        List of the top_k most similar text chunks
    """
    # [Step 1] Load default from config.yaml if not explicitly provided
    if top_k is None:
        top_k = cfg("retrieval", "vector_top_k")
    q_vec = embed_texts([query])[0]
    D, I = index.search(
        np.array([q_vec]).astype("float32"),
        top_k
    )
    return [documents[i] for i in I[0]]
