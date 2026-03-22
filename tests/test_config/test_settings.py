from config.settings import get


def test_get_section():
    """Config returns a dict for a known section."""
    ingestion = get("ingestion")
    assert isinstance(ingestion, dict)
    assert "chunk_size" in ingestion
    assert "chunk_overlap" in ingestion


def test_get_key():
    """Config returns a specific key value."""
    temp = get("llm", "temperature")
    assert isinstance(temp, (int, float))


def test_get_missing_section():
    """Unknown section returns empty dict."""
    result = get("nonexistent_section")
    assert result == {}


def test_get_missing_key():
    """Unknown key within a section returns None."""
    result = get("llm", "nonexistent_key")
    assert result is None


def test_llmops_monitoring_flags():
    """LLMOps monitoring section has expected flags."""
    monitoring = get("llmops", "monitoring")
    assert "track_token_usage" in monitoring
    assert "track_latency" in monitoring
    assert "log_prompts" in monitoring
