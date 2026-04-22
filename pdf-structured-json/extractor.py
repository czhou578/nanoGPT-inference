import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extracts plain text from a reasonably formatted PDF.
    
    Args:
        pdf_path: The path to the PDF file.
        
    Returns:
        The extracted raw text as a string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF missing at {pdf_path}")
        
    doc = fitz.open(pdf_path)
    text_content = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content.append(page.get_text())
        
    return "\n".join(text_content)
