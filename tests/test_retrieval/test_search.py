import numpy as np
from retrieval.search import search


def test_search_returns_results(monkeypatch):
    def fake_embed(texts):
        return [np.ones(10)]

    monkeypatch.setattr("retrieval.search.embed_texts", fake_embed)

    documents = ["doc1", "doc2", "doc3"]
    fake_index = type("FakeIndex", (), {
        "search": lambda self, x, k: (None, [[0]])
    })()

    results = search("test query", fake_index, documents, top_k=1)
    assert results == ["doc1"]


def test_search_top_k(monkeypatch):
    """Test that top_k controls the number of results."""
    def fake_embed(texts):
        return [np.ones(10)]

    monkeypatch.setattr("retrieval.search.embed_texts", fake_embed)

    documents = ["doc1", "doc2", "doc3"]
    fake_index = type("FakeIndex", (), {
        "search": lambda self, x, k: (None, [[0, 1, 2][:k]])
    })()

    results = search("test query", fake_index, documents, top_k=2)
    assert len(results) == 2
