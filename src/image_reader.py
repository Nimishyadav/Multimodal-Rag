import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_image_text(image_path):

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text
