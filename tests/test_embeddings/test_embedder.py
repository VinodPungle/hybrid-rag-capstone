from embeddings.embedder import embed_texts


def test_embed_texts_output_shape(monkeypatch):
    def fake_embedding(*args, **kwargs):
        class Fake:
            data = [type("obj", (), {"embedding": [0.1] * 1536})]
        return Fake()

    monkeypatch.setattr(
        "embeddings.embedder.client.embeddings.create",
        fake_embedding,
    )

    vectors = embed_texts(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536


def test_embed_texts_multiple(monkeypatch):
    """Test embedding multiple texts returns one vector per text."""
    call_count = 0

    def fake_embedding(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        class Fake:
            data = [type("obj", (), {"embedding": [0.1 * call_count] * 1536})]
        return Fake()

    monkeypatch.setattr(
        "embeddings.embedder.client.embeddings.create",
        fake_embedding,
    )

    vectors = embed_texts(["text one", "text two", "text three"])
    assert len(vectors) == 3
    assert all(len(v) == 1536 for v in vectors)
    # Each vector should be different since call_count increments
    assert vectors[0][0] != vectors[1][0]
