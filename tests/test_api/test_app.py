import pytest
from fastapi.testclient import TestClient
from api.app import app, _state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before each test."""
    _state["index"] = None
    _state["chunks"] = None
    _state["neo4j_driver"] = None
    yield


client = TestClient(app)


def test_health_no_document():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["document_loaded"] is False
    assert data["graph_available"] is False


def test_ask_without_document():
    resp = client.post("/ask", json={"query": "test question"})
    assert resp.status_code == 400
    assert "No document loaded" in resp.json()["detail"]


def test_ask_empty_query():
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422  # Pydantic validation error


def test_graph_no_driver():
    resp = client.get("/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []


def test_ingest_non_pdf():
    from io import BytesIO
    resp = client.post(
        "/ingest",
        files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]
