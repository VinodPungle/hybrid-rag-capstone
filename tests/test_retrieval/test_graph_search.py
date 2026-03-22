from retrieval.graph_search import graph_search


def test_graph_search_with_driver(monkeypatch):
    """Graph search delegates to query_graph_for_query."""
    monkeypatch.setattr(
        "retrieval.graph_search.query_graph_for_query",
        lambda driver, query, limit: ["A --[r]--> B"],
    )

    results = graph_search("test query", driver="fake_driver", limit=10)
    assert results == ["A --[r]--> B"]


def test_graph_search_no_driver(monkeypatch):
    """Graph search creates and closes its own driver when none provided."""
    closed = {"called": False}

    class FakeDriver:
        def close(self):
            closed["called"] = True

    monkeypatch.setattr(
        "retrieval.graph_search.get_driver",
        lambda: FakeDriver(),
    )
    monkeypatch.setattr(
        "retrieval.graph_search.query_graph_for_query",
        lambda driver, query, limit: [],
    )

    results = graph_search("test query", driver=None, limit=5)
    assert results == []
    assert closed["called"]
