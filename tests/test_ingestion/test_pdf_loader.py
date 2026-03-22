import os
import pytest
from dotenv import load_dotenv
from ingestion.pdf_loader import load_pdf_text

load_dotenv()


@pytest.mark.integration
def test_load_pdf_text_returns_string():
    input_file = os.getenv("INPUT_FILE")
    if not input_file or not os.path.exists(input_file):
        pytest.skip("INPUT_FILE not set or file not found")

    text = load_pdf_text(input_file)
    assert isinstance(text, str)
    assert len(text) > 0
