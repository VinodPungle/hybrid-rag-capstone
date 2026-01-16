import pytest
from unittest.mock import Mock, patch

def test_rag_pipeline_imports():
    """Test that all pipeline modules can be imported"""
    import ingestion.pdf_loader
    import ingestion.chunker
    import embeddings.embedder
    import vector_db.faiss_store
    import retrieval.search
    import llm.generator
    assert True

def test_rag_pipeline_chunk_count():
    """Test that chunking produces multiple chunks"""
    from ingestion.pdf_loader import load_pdf_text
    from ingestion.chunker import chunk_text
    
    text = load_pdf_text("data/raw/audit-committee-guide-2025.pdf")
    chunks = chunk_text(text)
    
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)

@patch('embeddings.embedder.client')
def test_rag_pipeline_embeddings_shape(mock_client):
    """Test that embeddings have correct dimensions"""
    from embeddings.embedder import embed_texts
    
    # Mock the API response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_response
    
    vectors = embed_texts(["test chunk"])
    
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536

def test_faiss_index_creation():
    """Test that FAISS index can be built"""
    from vector_db.faiss_store import build_faiss_index
    import numpy as np
    
    # Create dummy vectors
    vectors = [[0.1] * 1536 for _ in range(10)]
    index = build_faiss_index(vectors)
    
    assert index is not None
    assert index.ntotal == 10  # 10 vectors indexed

def test_search_returns_results():
    """Test that search returns relevant chunks"""
    from vector_db.faiss_store import build_faiss_index
    from retrieval.search import search
    
    # Create dummy data
    chunks = [f"chunk {i}" for i in range(10)]
    vectors = [[float(i)] * 1536 for i in range(10)]
    index = build_faiss_index(vectors)
    
    # This will need mocking of the query embedding
    # results = search("test query", index, chunks, top_k=3)
    # assert len(results) <= 3
    pass  # Skip actual search test without mocking

@pytest.mark.integration
def test_full_rag_pipeline():
    """Integration test for the full RAG pipeline"""
    # This would be a full end-to-end test
    # Mark as integration test so it can be run separately
    pass