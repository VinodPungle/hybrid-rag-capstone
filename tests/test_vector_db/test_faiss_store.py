import numpy as np
from vector_db.faiss_store import build_faiss_index

def test_faiss_index_builds():
    vectors = np.random.rand(5, 10).astype("float32")
    index = build_faiss_index(vectors)

    assert index.ntotal == 5
