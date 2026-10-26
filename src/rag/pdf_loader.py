from io import BytesIO
from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes, filename: str):
    """Extract text from a PDF while preserving page numbers."""

    reader = PdfReader(BytesIO(pdf_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "source": filename,
                "page": page_number,
                "text": text.strip()
            })

    return documents