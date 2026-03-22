"""
Text Chunking Module

Splits extracted document text into overlapping chunks for embedding.
Chunk size and overlap are driven by config.yaml → ingestion section.
"""

# [Step 1] Import config loader to read chunk_size and chunk_overlap from config.yaml
from config.settings import get as cfg


def chunk_text(text, chunk_size=None, overlap=None):
    """
    Split text into overlapping word-level chunks.

    Args:
        text: Raw text string to chunk
        chunk_size: Words per chunk (defaults to config.yaml → ingestion.chunk_size)
        overlap: Overlapping words between consecutive chunks
                 (defaults to config.yaml → ingestion.chunk_overlap)

    Returns:
        List of chunk strings
    """
    # [Step 1] Load defaults from config.yaml if not explicitly provided.
    # Uses 'is None' instead of 'or' so that 0 can be passed as a valid value.
    _ing = cfg("ingestion")
    if chunk_size is None:
        chunk_size = _ing["chunk_size"]
    if overlap is None:
        overlap = _ing["chunk_overlap"]

    words = text.split()
    chunks = []
    start = 0

    # [Step 3] Guard against infinite loop: ensure step is always >= 1,
    # even if overlap >= chunk_size (bug fix discovered during pytest improvements)
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += step

    return chunks
