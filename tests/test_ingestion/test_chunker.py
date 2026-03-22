from ingestion.chunker import chunk_text


def test_chunk_text_creates_chunks():
    text = "word " * 2000
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_text_with_overlap():
    """Test that overlap produces more chunks than no overlap."""
    text = "word " * 200
    chunks_with_overlap = chunk_text(text, chunk_size=50, overlap=10)
    chunks_no_overlap = chunk_text(text, chunk_size=50, overlap=0)

    assert len(chunks_with_overlap) > len(chunks_no_overlap)


def test_chunk_text_empty_input():
    """Test with empty string."""
    chunks = chunk_text("", chunk_size=100, overlap=20)
    assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")


def test_chunk_text_small_input():
    """Test with text smaller than chunk size."""
    text = "small text"
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_uses_config_defaults():
    """Test that defaults come from config when not specified."""
    text = "word " * 2000
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
