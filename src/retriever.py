import pickle
import difflib
import os
import re
import numpy as np

from src.chunker import chunk_segment
from src.embedder import create_embeddings
from src.excel_reader import compare_workbooks, read_excel
from src.vector_store import VectorStore
from src.knowledge_graph import load_graph, query_graph
from src.reranker import rerank as cross_encoder_rerank


SUMMARY_TRIGGER_WORDS = [
    "summary", "summarize", "summarise", "explain", "read",
    "overview", "describe", "tell me about", "what is this about",
    "what does", "what's in", "what is in", "contents of"
]
DOC_WORDS = [
    "pdf", "document", "doc", "file", "image", "picture", "diagram", "photo",
    "excel", "spreadsheet", "workbook", "worksheet", "sheet", "table", "row",
]

STATS_KEYWORDS = [
    "how many table", "how many image", "how many page",
    "number of table", "number of image", "number of page",
    "count of table", "count of image", "count of page"
]

GRAPH_KEYWORDS = [
    "relationship", "relationships", "related to", "connection between",
    "connections", "entities", "entity", "connected to", "linked to",
    "knowledge graph", "belongs to", "how is", "how are"
]


def is_graph_query(query):
    q = query.lower()
    return any(keyword in q for keyword in GRAPH_KEYWORDS)


def load_index_and_chunks(index_path="storage/index.faiss", chunks_path="storage/chunks.pkl"):
    index = VectorStore.load(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def is_summary_query(query):
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
        if stats.get("sheets"):
            lines[-1] += f", {stats['sheets']} worksheet(s)"
        total_pages += stats["pages"]
        total_tables += stats["tables"]
        total_images += stats["images"]

    header = (
        f"Across all documents: {total_pages} page(s), "
        f"{total_tables} table(s), {total_images} image(s) total.\n"
    )
    return header + "\n".join(lines)


def get_known_sources():
    _, chunks = load_index_and_chunks()
    return sorted(set(c["source"] for c in chunks))


def find_mentioned_source(query, sources):
    q_words = query.lower().replace(".", " ").replace("_", " ").replace("-", " ").split()

    best_match = None
    best_score = 0

    for src in sources:
        name_part = src.lower()
        for ext in (".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".xls"):
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


def is_excel_relationship_query(query):
    """Detect relationship questions that explicitly target workbooks."""
    query_text = query.lower()
    has_excel_term = any(
        term in query_text
        for term in ("excel", "spreadsheet", "workbook", "worksheet", ".xlsx", ".xls")
    )
    has_relationship_term = any(
        term in query_text
        for term in ("relationship", "related", "connection", "compare", "difference", "link")
    )
    return has_excel_term and has_relationship_term


def is_excel_query(query):
    """Detect questions that ask about spreadsheet content."""
    query_text = query.lower()
    return any(
        term in query_text
        for term in ("excel", "spreadsheet", "workbook", "worksheet", ".xlsx", ".xls")
    )


def asks_for_all_excel_files(query):
    """Return true for plural workbook requests instead of one-file requests."""
    query_text = query.lower()
    return any(
        phrase in query_text
        for phrase in (
            "excel files", "all excel", "workbooks", "spreadsheets",
            ".xlsx files", ".xls files",
        )
    )


def get_excel_chunks():
    """Read every current workbook so Excel answers never use stale storage."""
    excel_chunks = []
    data_folder = "data"
    if os.path.isdir(data_folder):
        for filename in sorted(os.listdir(data_folder)):
            if (
                not filename.lower().endswith((".xlsx", ".xls"))
                or filename.startswith("~$")
            ):
                continue
            segments, _ = read_excel(os.path.join(data_folder, filename))
            for segment_index, segment in enumerate(segments):
                segment.setdefault("id", f"{filename}:segment:{segment_index}")
                excel_chunks.extend(chunk_segment(segment, filename, segment_index))

    if excel_chunks:
        return excel_chunks

    # Keep a useful fallback if the data folder is temporarily unavailable.
    try:
        _, chunks = load_index_and_chunks()
    except (FileNotFoundError, OSError):
        return []
    excel_chunks = [
        chunk for chunk in chunks
        if chunk.get("source", "").lower().endswith((".xlsx", ".xls"))
    ]
    return excel_chunks


def build_excel_relationship_context(chunks):
    """Create explicit workbook/sheet/row context for relationship answers."""
    if not chunks:
        return "No Excel workbook chunks are present in the rebuilt index."

    lines = []
    workbook_paths = [
        os.path.join("data", filename)
        for filename in sorted(os.listdir("data"))
        if (
            filename.lower().endswith((".xlsx", ".xls"))
            and not filename.startswith("~$")
        )
    ]
    if workbook_paths:
        lines.append(compare_workbooks(workbook_paths))
    lines.append("EXCEL WORKBOOK EVIDENCE:")
    for chunk in chunks:
        location = chunk["source"]
        if chunk.get("sheet"):
            location += f" | sheet: {chunk['sheet']}"
        if chunk.get("row_start") is not None:
            location += f" | rows: {chunk['row_start']}-{chunk.get('row_end', chunk['row_start'])}"
        lines.append(f"[{location}]\n{chunk['text']}")
    return "\n\n".join(lines)


def _vector_scores(distances):
    """Converts FAISS L2 distances into a 0-1 similarity-like score
    (smaller distance = higher score)."""
    distances = np.array(distances, dtype="float32")
    max_d = distances.max() if len(distances) > 0 and distances.max() > 0 else 1.0
    return 1 - (distances / max_d)


def _terms(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _lexical_score(query, text):
    query_terms = _terms(query)
    text_terms = _terms(text)
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms.intersection(text_terms)) / len(query_terms)


def search(query, top_k_per_source=4, search_pool=None, use_reranker=True):
    """Retrieve evidence while guaranteeing cross-source coverage."""
    index, chunks = load_index_and_chunks()

    query_embedding = create_embeddings([query])[0]
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

    pool_size = len(chunks) if search_pool is None else min(search_pool, len(chunks))
    distances, indices = index.search(query_embedding, pool_size)
    valid_candidates = [
        (chunks[i], distance)
        for i, distance in zip(indices[0], distances[0])
        if 0 <= i < len(chunks)
    ]
    candidate_chunks = [candidate for candidate, _ in valid_candidates]
    vector_scores = _vector_scores([distance for _, distance in valid_candidates])

    # --- Graph score: does this chunk's SOURCE have a graph fact relevant to the query? ---
    try:
        triples = load_graph()
        matching_triples = query_graph(query, triples)
        graph_sources = {t["source"] for t in matching_triples}
        graph_chunk_ids = {
            t.get("chunk_id") for t in matching_triples if t.get("chunk_id")
        }
    except FileNotFoundError:
        graph_sources = set()
        graph_chunk_ids = set()

    scored_candidates = []
    for chunk, v_score in zip(candidate_chunks, vector_scores):
        lexical_score = _lexical_score(query, chunk["text"])
        graph_score = 1.0 if (
            chunk.get("id") in graph_chunk_ids
            or chunk["source"] in graph_sources
        ) else 0.0
        final_score = (0.55 * float(v_score)) + (0.25 * lexical_score) + (0.20 * graph_score)
        enriched_chunk = dict(chunk)
        enriched_chunk["retrieval_score"] = round(float(final_score), 4)
        enriched_chunk["evidence"] = {
            "vector": round(float(v_score), 4),
            "lexical": round(float(lexical_score), 4),
            "graph": round(float(graph_score), 4),
        }
        scored_candidates.append((enriched_chunk, final_score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # --- Per-source diversity on top of the fused ranking (cross-document reasoning) ---
    final_results = []
    seen_per_source = {}
    for chunk, score in scored_candidates:
        src = chunk["source"]
        seen_per_source.setdefault(src, 0)
        if seen_per_source[src] < top_k_per_source:
            final_results.append(chunk)
            seen_per_source[src] += 1

    # --- Reranking pass on the final shortlist ---
    if use_reranker and final_results:
        final_results = cross_encoder_rerank(query, final_results, top_k=len(final_results))

    return final_results


def build_grounded_context(query, results=None, max_graph_facts=12):
    """Build the multimodal evidence packet passed to the answer model."""
    results = results if results is not None else search(query)
    try:
        graph_facts = query_graph(query, load_graph())
    except FileNotFoundError:
        graph_facts = []

    evidence_lines = []
    for result in results:
        location = result.get("source", "unknown source")
        if result.get("page") is not None:
            location += f", page {result['page']}"
        if result.get("sheet"):
            location += f", sheet {result['sheet']}"
        if result.get("row_start") is not None:
            row_end = result.get("row_end") or result["row_start"]
            location += f", rows {result['row_start']}-{row_end}"
        modality = result.get("modality", result.get("type", "TEXT"))
        evidence_lines.append(
            f"[{modality} | {location} | chunk {result.get('id', 'unknown')}]\n"
            f"{result['text']}"
        )

    graph_lines = [
        f"{fact['subject']} --[{fact['relation']}]--> {fact['object']} "
        f"(source: {fact['source']})"
        for fact in graph_facts[:max_graph_facts]
    ]

    sections = []
    if evidence_lines:
        sections.append("RETRIEVED MULTIMODAL EVIDENCE:\n" + "\n\n".join(evidence_lines))
    if graph_lines:
        sections.append("KNOWLEDGE GRAPH FACTS:\n" + "\n".join(graph_lines))
    return "\n\n".join(sections) or "No relevant evidence was retrieved."