import numpy as np
from retrieval.hybrid_search import hybrid_search


def test_hybrid_search_vector_only(monkeypatch):
    """Hybrid search works when graph search returns nothing."""
    def fake_embed(texts):
        return [np.ones(10)]

    monkeypatch.setattr("retrieval.search.embed_texts", fake_embed)
    monkeypatch.setattr(
        "retrieval.hybrid_search.graph_search",
        lambda query, driver, limit: [],
    )

    chunks = ["chunk about auditing", "chunk about finance", "chunk about risk"]
    fake_index = type("FakeIndex", (), {
        "search": lambda self, x, k: (None, [[0, 1, 2][:k]])
    })()

    result = hybrid_search("audit", fake_index, chunks, driver=None, top_k=2)

    assert "vector_results" in result
    assert "graph_results" in result
    assert "combined_context" in result
    assert len(result["vector_results"]) == 2
    assert result["graph_results"] == []


def test_hybrid_search_combined_context(monkeypatch):
    """Combined context includes both vector and graph results."""
    def fake_embed(texts):
        return [np.ones(10)]

    monkeypatch.setattr("retrieval.search.embed_texts", fake_embed)
    monkeypatch.setattr(
        "retrieval.hybrid_search.graph_search",
        lambda query, driver, limit: ["Entity A --[relates_to]--> Entity B"],
    )

    chunks = ["chunk one"]
    fake_index = type("FakeIndex", (), {
        "search": lambda self, x, k: (None, [[0]])
    })()

    result = hybrid_search("test", fake_index, chunks, driver="fake_driver", top_k=1)

    assert len(result["vector_results"]) == 1
    assert len(result["graph_results"]) == 1
    assert "Document Context" in result["combined_context"]
    assert "Knowledge Graph Context" in result["combined_context"]
