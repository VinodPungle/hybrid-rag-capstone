import json
from graph_db.entity_extractor import extract_entities_and_relationships, extract_graph_from_chunks


def _make_mock_response(entities, relationships):
    """Create a mock OpenAI response containing JSON entity/relationship data."""
    payload = json.dumps({"entities": entities, "relationships": relationships})

    class Usage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150

    class Message:
        content = payload

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]
        usage = Usage()

    return Response()


def test_extract_entities_and_relationships(monkeypatch):
    entities = [{"name": "Audit Committee", "type": "Committee"}]
    relationships = [{"source": "Audit Committee", "target": "Board", "relation": "reports_to"}]

    monkeypatch.setattr(
        "graph_db.entity_extractor.client.chat.completions.create",
        lambda *args, **kwargs: _make_mock_response(entities, relationships),
    )

    ents, rels = extract_entities_and_relationships("The audit committee reports to the board.")
    assert len(ents) == 1
    assert ents[0]["name"] == "Audit Committee"
    assert len(rels) == 1
    assert rels[0]["relation"] == "reports_to"


def test_extract_graph_from_chunks(monkeypatch):
    entities = [{"name": "Entity A", "type": "Concept"}]
    relationships = []

    monkeypatch.setattr(
        "graph_db.entity_extractor.client.chat.completions.create",
        lambda *args, **kwargs: _make_mock_response(entities, relationships),
    )

    chunks = ["chunk one", "chunk two"]
    all_ents, all_rels = extract_graph_from_chunks(chunks)

    assert len(all_ents) == 2  # one entity per chunk
    assert all_ents[0]["source_chunk"] == 0
    assert all_ents[1]["source_chunk"] == 1


def test_extract_handles_markdown_fenced_json(monkeypatch):
    """LLMs sometimes wrap JSON in ```json ... ``` fences."""
    payload = '```json\n{"entities": [{"name": "X", "type": "Concept"}], "relationships": []}\n```'

    class Usage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150

    class Message:
        content = payload

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]
        usage = Usage()

    monkeypatch.setattr(
        "graph_db.entity_extractor.client.chat.completions.create",
        lambda *args, **kwargs: Response(),
    )

    ents, rels = extract_entities_and_relationships("Some text")
    assert len(ents) == 1
    assert ents[0]["name"] == "X"
