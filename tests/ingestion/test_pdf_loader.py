from ingestion.pdf_loader import load_pdf_text

def test_load_pdf_text_returns_string():
    # Use forward slashes (works on Windows and Unix)
    text = load_pdf_text("data/raw/audit-committee-guide-2025.pdf")
    assert isinstance(text, str)
    assert len(text) > 0
