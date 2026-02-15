import os
from dotenv import load_dotenv
from ingestion.pdf_loader import load_pdf_text
load_dotenv()

def test_load_pdf_text_returns_string():
    # Use forward slashes (works on Windows and Unix)
    text = load_pdf_text(os.getenv("INPUT_FILE"))
    assert isinstance(text, str)
    assert len(text) > 0
