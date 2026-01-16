import faiss
import numpy as np

def build_faiss_index(vectors):
    """
    Build a FAISS index from embedding vectors.
    
    Args:
        vectors: List of embedding vectors or numpy array
        
    Returns:
        FAISS index
    """
    # Convert to numpy array if it's a list
    if isinstance(vectors, list):
        vectors = np.array(vectors, dtype=np.float32)
    elif not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors, dtype=np.float32)
    
    # Ensure correct dtype
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    # Get dimension
    dimension = vectors.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    return index
