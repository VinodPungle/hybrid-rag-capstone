from pypdf import PdfReader

def load_pdf_text(pdf_path: str) -> str:
    """
    Load text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    reader = PdfReader(pdf_path)  # Use the parameter, not hardcoded path
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text