import numpy as np
from embeddings.embedder import embed_texts

def search(query, index, documents, top_k=3):
    q_vec = embed_texts([query])[0]
    D, I = index.search(
        np.array([q_vec]).astype("float32"),
        top_k
    )
    return [documents[i] for i in I[0]]
