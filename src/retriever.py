import pickle
import numpy as np

from src.embedder import create_embeddings
from src.vector_store import VectorStore


SUMMARY_TRIGGER_WORDS = [
    "summary", "summarize", "summarise", "explain",
    "overview", "describe", "tell me about", "what is this about"
]
DOC_WORDS = ["pdf", "document", "doc", "file"]

STATS_KEYWORDS = [
    "how many table", "how many image", "how many page",
    "number of table", "number of image", "number of page",
    "count of table", "count of image", "count of page"
]


def load_index_and_chunks(index_path="storage/index.faiss", chunks_path="storage/chunks.pkl"):
    index = VectorStore.load(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def is_summary_query(query):
    """Catches 'explain/summarize/overview of the pdf/document' style questions,
    even if worded loosely (e.g. 'explain the sample pdf')."""
    q = query.lower()
    has_trigger = any(w in q for w in SUMMARY_TRIGGER_WORDS)
    has_doc_word = any(w in q for w in DOC_WORDS)
    return has_trigger and has_doc_word


def is_stats_query(query):
    q = query.lower()
    return any(keyword in q for keyword in STATS_KEYWORDS)


def get_stats_answer(stats_path="storage/stats.pkl"):
    with open(stats_path, "rb") as f:
        stats_by_source = pickle.load(f)

    total_tables = 0
    total_images = 0
    total_pages = 0
    lines = []

    for source, stats in stats_by_source.items():
        lines.append(
            f"- {source}: {stats['pages']} page(s), "
            f"{stats['tables']} table(s), {stats['images']} image(s)"
        )
        total_pages += stats["pages"]
        total_tables += stats["tables"]
        total_images += stats["images"]

    header = (
        f"Across all documents: {total_pages} page(s), "
        f"{total_tables} table(s), {total_images} image(s) total.\n"
    )
    return header + "\n".join(lines)


def get_known_sources():
    """Returns the list of all filenames currently indexed."""
    _, chunks = load_index_and_chunks()
    return sorted(set(c["source"] for c in chunks))


import difflib


def find_mentioned_source(query, sources):
    """Checks if the question names a specific file — tolerant of small typos
    (e.g. 'simple.pdf' still matches the real file 'sample pdf.pdf')."""
    q_words = query.lower().replace(".", " ").replace("_", " ").replace("-", " ").split()

    best_match = None
    best_score = 0

    for src in sources:
        name_part = src.lower()
        for ext in (".pdf", ".png", ".jpg", ".jpeg"):
            name_part = name_part.replace(ext, "")
        name_words = name_part.replace("-", " ").replace("_", " ").split()

        if not name_words:
            continue

        matched = 0
        for word in name_words:
            close = difflib.get_close_matches(word, q_words, n=1, cutoff=0.7)
            if close:
                matched += 1

        score = matched / len(name_words)
        if score > best_score:
            best_score = score
            best_match = src

    return best_match if best_score >= 0.5 else None


def get_summary_chunks(per_source_limit=8, source_filter=None):
    """Evenly-spaced chunks for a whole-document summary.
    If source_filter is given, only that ONE file's chunks are used."""
    _, chunks = load_index_and_chunks()

    if source_filter:
        chunks = [c for c in chunks if c["source"] == source_filter]

    grouped = {}
    for c in chunks:
        grouped.setdefault(c["source"], []).append(c)

    summary_chunks = []
    for source, source_chunks in grouped.items():
        total = len(source_chunks)
        if total <= per_source_limit:
            selected = source_chunks
        else:
            step = total / per_source_limit
            indices = [int(i * step) for i in range(per_source_limit)]
            selected = [source_chunks[i] for i in indices]
        summary_chunks.extend(selected)

    return summary_chunks


def search(query, top_k_per_source=2, search_pool=20):
    index, chunks = load_index_and_chunks()

    query_embedding = create_embeddings([query])[0]
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_embedding, search_pool)
    all_results = [chunks[i] for i in indices[0]]

    final_results = []
    seen_per_source = {}

    for r in all_results:
        src = r["source"]
        seen_per_source.setdefault(src, 0)
        if seen_per_source[src] < top_k_per_source:
            final_results.append(r)
            seen_per_source[src] += 1

    return final_results