from pypdf import PdfReader
import os

def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and returns the extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text.strip()
