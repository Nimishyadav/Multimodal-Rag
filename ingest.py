import os
import pickle

from src.pdf_reader import read_pdf
from src.image_reader import read_image_segments

from src.excel_reader import (
    read_excel,
    workbook_relationship_triples,
    excel_entity_triples
)

from src.chunker import chunk_segment
from src.embedder import create_embeddings
from src.vector_store import VectorStore

from src.knowledge_graph import (
    build_graph,
    save_graph,
    add_structural_triples
)

DATA_FOLDER = "data"
EXCEL_EXTENSIONS = (".xlsx", ".xls")
PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

all_files = sorted(os.listdir(DATA_FOLDER))

input_files = [
    f for f in all_files
    if f.lower().endswith(
        EXCEL_EXTENSIONS + PDF_EXTENSIONS + IMAGE_EXTENSIONS
    )
    and not f.startswith("~$")
]

if not input_files:
    print("No supported files found in data/ folder.")
    exit()

print(f"Found {len(input_files)} file(s) in data/")

all_chunks = []
stats_by_source = {}
sources_text = {}
structural_counts = {}


"""READ ALL SUPPORTED FILES"""


for filename in input_files:

    path = os.path.join(DATA_FOLDER, filename)

    extension = os.path.splitext(filename)[1].lower()
    print(f"Reading {extension[1:].upper()}: {filename}...")

    if extension in EXCEL_EXTENSIONS:
        segments, stats = read_excel(path)
    elif extension in PDF_EXTENSIONS:
        segments, stats = read_pdf(path)
    else:
        segments = read_image_segments(path)
        stats = {
            "pages": 0,
            "tables": 0,
            "images": 1,
        }

    stats_by_source[filename] = stats

    sources_text[filename] = "\n".join(
        s["text"]
        for s in segments
    )

    modality_counts = {"TABLE": 0, "IMAGE": 0, "TEXT": 0}

    for segment_index, seg in enumerate(segments):

        seg.setdefault(
            "id",
            f"{filename}:segment:{segment_index}"
        )

        seg.setdefault(
            "modality",
            seg["type"]
        )

        chunks = chunk_segment(
            seg,
            filename,
            segment_index
        )

        all_chunks.extend(chunks)

        modality = seg.get("modality", seg.get("type", "TEXT")).upper()
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    structural_counts[filename] = modality_counts


"""EMBEDDINGS"""


print(f"Total chunks created: {len(all_chunks)}")

texts_only = [
    chunk["text"]
    for chunk in all_chunks
]

if not texts_only:
    raise RuntimeError("No readable content was found in data/.")

embeddings = create_embeddings(texts_only)

store = VectorStore(
    len(embeddings[0])
)

store.add(embeddings)

store.save(
    "storage/index.faiss"
)


"""SAVE DATA"""


with open(
    "storage/chunks.pkl",
    "wb"
) as f:
    pickle.dump(
        all_chunks,
        f
    )

with open(
    "storage/stats.pkl",
    "wb"
) as f:
    pickle.dump(
        stats_by_source,
        f
    )

with open(
    "storage/source_text.pkl",
    "wb"
) as f:
    pickle.dump(
        sources_text,
        f
    )


"""KNOWLEDGE GRAPH"""


print("Building knowledge graph...")

triples = build_graph(
    sources_text,
    all_chunks
)

triples += add_structural_triples(
    structural_counts
)

excel_paths = [
    os.path.join(DATA_FOLDER, filename)
    for filename in input_files
    if filename.lower().endswith(EXCEL_EXTENSIONS)
]

triples += workbook_relationship_triples(
    excel_paths
)

triples += excel_entity_triples(
    excel_paths
)

save_graph(
    triples,
    source_files=input_files,
    chunks=all_chunks,
    stats_by_source=stats_by_source,
    source_text_by_source=sources_text,
)

print(
    f"Knowledge graph saved with {len(triples)} relationships."
)

print(
    "Multimodal ingestion completed successfully!"
)