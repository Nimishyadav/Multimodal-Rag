import io
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def read_pdf(pdf_path):
    """Extracts normal text, tables, AND text inside embedded images (via OCR).
    Also returns stats: number of pages, tables, and images found."""
    full_text = ""
    table_count = 0
    image_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

            tables = page.extract_tables()
            for table in tables:
                table_count += 1
                full_text += "\n[TABLE]\n"
                for row in table:
                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                    full_text += row_text + "\n"
                full_text += "[/TABLE]\n"

    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        image_count += len(images)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            pil_image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(pil_image)

            if ocr_text.strip():
                full_text += "\n[IMAGE TEXT]\n" + ocr_text + "\n[/IMAGE TEXT]\n"

    doc.close()

    stats = {
        "pages": page_count,
        "tables": table_count,
        "images": image_count
    }

    return full_text, stats