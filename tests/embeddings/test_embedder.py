from embeddings.embedder import embed_texts

def test_embed_texts_output_shape(monkeypatch):

    def fake_embedding(*args, **kwargs):
        class Fake:
            data = [type("obj", (), {"embedding": [0.1]*1536})]
        return Fake()

    monkeypatch.setattr(
        "embeddings.embedder.client.embeddings.create",
        fake_embedding
    )

    vectors = embed_texts(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536

    
