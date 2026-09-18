import os
import pickle

from src.pdf_reader import read_pdf
from src.image_reader import read_image_text
from src.chunker import chunk_text
from src.embedder import create_embeddings
from src.vector_store import VectorStore

DATA_FOLDER = "data"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

all_files = os.listdir(DATA_FOLDER)
pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]

if not pdf_files and not image_files:
    print("No PDFs or images found in the data/ folder.")
    exit()

print(f"Found {len(pdf_files)} PDF(s) and {len(image_files)} image(s)")

all_chunks = []       # {"text": ..., "source": filename} for every chunk
stats_by_source = {}  # {"filename": {"pages": X, "tables": X, "images": X}}

for filename in pdf_files:
    path = os.path.join(DATA_FOLDER, filename)
    print(f"Reading PDF: {filename}...")

    text, stats = read_pdf(path)
    stats_by_source[filename] = stats

    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})

for filename in image_files:
    path = os.path.join(DATA_FOLDER, filename)
    print(f"Reading image (OCR): {filename}...")

    text = read_image_text(path)
    stats_by_source[filename] = {"pages": 1, "tables": 0, "images": 1}

    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})

print(f"Total chunks created: {len(all_chunks)}")

texts_only = [c["text"] for c in all_chunks]
embeddings = create_embeddings(texts_only)

store = VectorStore(len(embeddings[0]))
store.add(embeddings)
store.save("storage/index.faiss")

with open("storage/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

with open("storage/stats.pkl", "wb") as f:
    pickle.dump(stats_by_source, f)

print("Index and chunks saved successfully!")