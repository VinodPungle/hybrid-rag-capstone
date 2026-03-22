import pytest


def test_rag_pipeline_imports():
    """Test that all pipeline modules can be imported."""
    import ingestion.pdf_loader
    import ingestion.chunker
    import embeddings.embedder
    import vector_db.faiss_store
    import retrieval.search
    import retrieval.hybrid_search
    import retrieval.graph_search
    import graph_db.entity_extractor
    import graph_db.neo4j_store
    import llm.generator
    import config.settings
    import utils.logger


def test_config_loads_all_sections():
    """Test that config has all expected top-level sections."""
    from config.settings import get
    for section in ["ingestion", "embedding", "llm", "entity_extraction",
                    "retrieval", "neo4j", "ui", "llmops"]:
        result = get(section)
        assert isinstance(result, dict), f"Section '{section}' missing or not a dict"


@pytest.mark.integration
def test_chunking_from_pdf():
    """Integration test: load PDF and chunk it."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    input_file = os.getenv("INPUT_FILE")
    if not input_file or not os.path.exists(input_file):
        pytest.skip("INPUT_FILE not set or file not found")

    from ingestion.pdf_loader import load_pdf_text
    from ingestion.chunker import chunk_text

    text = load_pdf_text(input_file)
    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
