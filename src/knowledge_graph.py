import json
import os
import pickle
import re
from collections import defaultdict
import ollama

MODEL_NAME = "llama3"  


def _clean_entity_name(name):
    """Removes leading numbering like '1. ' or '12) ' from an entity name,
    so the SAME entity mentioned multiple times becomes ONE node in the
    graph instead of fragmenting into '1. X', '2. X', '3. X'."""
    name = name.strip()
    name = re.sub(r'^\d+[\.\)]\s*', '', name)
    return name.strip()


def extract_triples(text, source, metadata=None):
    """Uses the LLM to pull out simple (subject, relation, object) facts from text.
    This is the 'knowledge graph' building block — each triple is one fact."""
    prompt = f"""Extract factual relationships from the text below as simple triples.
Format each one EXACTLY as: subject | relation | object
One triple per line. Only extract clear, factual relationships. Maximum 15 triples.

Text:
{text[:3000]}

Triples:
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 400}
    )

    raw = response["message"]["content"]
    triples = []

    for line in raw.split("\n"):
        parts = line.split("|")
        if len(parts) == 3:
            subject, relation, obj = [p.strip() for p in parts]
            subject = _clean_entity_name(subject)
            obj = _clean_entity_name(obj)
            if subject and relation and obj:
                triple = {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "source": source
                }
                if metadata:
                    triple.update(metadata)
                triples.append(triple)

    return triples


def extract_image_relationship_triples(desc1, id1, desc2, id2):
    """Compares two image descriptions and extracts their relationship as
    REAL triples (subject | relation | object) — using id1/id2 as names —
    so this becomes actual queryable graph data, not a one-off free-text answer."""
    prompt = f"""Compare these two image descriptions and extract relationships
between them as simple triples. Format each one EXACTLY as: subject | relation | object
Use "{id1}" and "{id2}" as the subject/object names in your triples.
Only extract relationships clearly supported by the descriptions. Maximum 5 triples.

{id1} description:
{desc1}

{id2} description:
{desc2}

Triples:
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 300}
    )

    raw = response["message"]["content"]
    triples = []

    for line in raw.split("\n"):
        parts = line.split("|")
        if len(parts) == 3:
            subject, relation, obj = [p.strip() for p in parts]
            subject = _clean_entity_name(subject)
            obj = _clean_entity_name(obj)
            if subject and relation and obj:
                triples.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "source": "cross-image"
                })

    return triples


def build_graph(sources_text, chunks=None):
    """sources_text: {filename: full_text}. Returns a list of ALL triples
    extracted across every document/image."""
    all_triples = []

    chunks_by_source = {}
    for chunk in chunks or []:
        chunks_by_source.setdefault(chunk["source"], []).append(chunk)

    for source, text in sources_text.items():
        print(f"Extracting relationships from {source}...")
        source_chunks = chunks_by_source.get(source, [])
        modality_summary = ", ".join(
            f"[{chunk.get('modality', chunk.get('type', 'TEXT'))}] {chunk['text']}"
            for chunk in source_chunks[:40]
        )
        graph_text = modality_summary or text
        triples = extract_triples(graph_text, source)
        for triple in triples:
            matching_chunk = next(
                (
                    chunk for chunk in source_chunks
                    if triple["subject"].lower() in chunk["text"].lower()
                    or triple["object"].lower() in chunk["text"].lower()
                ),
                None,
            )
            if matching_chunk:
                triple["chunk_id"] = matching_chunk.get("id")
                triple["modality"] = matching_chunk.get(
                    "modality", matching_chunk.get("type", "TEXT")
                )
        all_triples.extend(triples)

    return all_triples


def save_graph(
    triples,
    path="storage/graph.pkl",
    source_files=None,
    chunks=None,
    stats_by_source=None,
    source_text_by_source=None,
):
    """Save graph relationships and extracted content in readable formats."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if chunks is None:
        try:
            with open("storage/chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
        except (FileNotFoundError, OSError, pickle.PickleError):
            chunks = []
    if stats_by_source is None:
        try:
            with open("storage/stats.pkl", "rb") as f:
                stats_by_source = pickle.load(f)
        except (FileNotFoundError, OSError, pickle.PickleError):
            stats_by_source = {}
    if source_text_by_source is None:
        try:
            with open("storage/source_text.pkl", "rb") as f:
                source_text_by_source = pickle.load(f)
        except (FileNotFoundError, OSError, pickle.PickleError):
            source_text_by_source = {}

    source_set = set(source_files or [])
    selected_chunks = [
        chunk for chunk in chunks
        if not source_set or chunk.get("source") in source_set
    ]

    with open(path, "wb") as f:
        pickle.dump(triples, f)

    sources = sorted(
        set(source_files or [])
        | {triple.get("source", "") for triple in triples if triple.get("source")}
    )
    graph_data = {
        "source_files": sources,
        "stats_by_source": stats_by_source,
        "source_text": source_text_by_source,
        "relationship_count": len(triples),
        "chunk_count": len(selected_chunks),
        "chunks": selected_chunks,
        "relationships": triples,
    }

    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    text_path = os.path.splitext(path)[0] + "_readable.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        chunks_by_source = defaultdict(list)
        relationships_by_source = defaultdict(list)
        for chunk in selected_chunks:
            chunks_by_source[chunk.get("source", "unknown source")].append(chunk)
        for triple in triples:
            relationships_by_source[triple.get("source", "unknown source")].append(triple)

        report_sources = sorted(
            set(sources)
            | set(chunks_by_source)
            | set(relationships_by_source)
        )
        f.write("KNOWLEDGE GRAPH - READABLE DATA REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write("HOW TO READ THIS FILE\n")
        f.write("- Each SOURCE section contains data from one input file.\n")
        f.write("- EXTRACTED CONTENT is what the readers found in that file.\n")
        f.write("- RELATIONSHIPS are graph edges: subject -> relation -> object.\n")
        f.write(f"\nTOTAL SOURCE FILES: {len(report_sources)}\n")
        f.write(f"TOTAL CONTENT CHUNKS: {len(selected_chunks)}\n")
        f.write(f"TOTAL RELATIONSHIPS: {len(triples)}\n\n")

        for source_index, source in enumerate(report_sources, start=1):
            stats = stats_by_source.get(source, {})
            f.write(f"\n{'#' * 80}\n")
            f.write(f"SOURCE {source_index}: {source}\n")
            f.write(f"{'#' * 80}\n")
            if stats:
                f.write("FILE STATISTICS\n")
                for key, value in stats.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            f.write("FULL SOURCE TEXT (VERBATIM)\n")
            f.write("-" * 28 + "\n")
            full_source_text = source_text_by_source.get(source, "").strip()
            f.write(
                f"{full_source_text}\n"
                if full_source_text
                else "No verbatim source text stored for this source.\n"
            )

            source_chunks = chunks_by_source.get(source, [])
            f.write(f"EXTRACTED CONTENT ({len(source_chunks)} chunks)\n")
            f.write("-" * 24 + "\n")
            if source_chunks:
                for chunk_index, chunk in enumerate(source_chunks, start=1):
                    location = []
                    for key in ("page", "sheet", "row_start", "row_end"):
                        if chunk.get(key) is not None:
                            location.append(f"{key}={chunk[key]}")
                    modality = chunk.get("modality", chunk.get("type", "TEXT"))
                    location_text = f" | {', '.join(location)}" if location else ""
                    f.write(f"\nCONTENT {chunk_index} [{modality}{location_text}]\n")
                    f.write(f"ID: {chunk.get('id', 'unknown')}\n")
                    f.write(f"{chunk.get('text', '').strip()}\n")
            else:
                f.write("No extracted content stored for this source.\n")

            source_relationships = relationships_by_source.get(source, [])
            f.write(f"\nRELATIONSHIPS ({len(source_relationships)})\n")
            f.write("-" * 16 + "\n")
            if source_relationships:
                for relationship_index, triple in enumerate(source_relationships, start=1):
                    f.write(
                        f"{relationship_index}. SUBJECT: {triple.get('subject', '')}\n"
                        f"   RELATION: {triple.get('relation', '')}\n"
                        f"   OBJECT: {triple.get('object', '')}\n"
                    )
                    if triple.get("modality"):
                        f.write(f"   MODALITY: {triple['modality']}\n")
                    if triple.get("details"):
                        f.write(f"   DETAILS: {triple['details']}\n")
            else:
                f.write("No relationships extracted for this source.\n")


def load_graph(path="storage/graph.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def add_structural_triples(structural_counts):
    """Adds automatic 'belongs_to' relationships for every table/image,
    e.g. 'Image_1_report.pdf belongs_to report.pdf'.
    No LLM call needed — this is deterministic, based on what we already
    know from parsing (how many tables/images each document has)."""
    triples = []

    for source, counts in structural_counts.items():
        for i in range(1, counts.get("TABLE", 0) + 1):
            triples.append({
                "subject": f"Table_{i}_{source}",
                "relation": "belongs_to",
                "object": source,
                "source": source
            })
        for i in range(1, counts.get("IMAGE", 0) + 1):
            triples.append({
                "subject": f"Image_{i}_{source}",
                "relation": "belongs_to",
                "object": source,
                "source": source
            })

    return triples


def query_graph(query, triples):
    """Finds triples where the subject or object is mentioned in the question.
    This is the 'multi-hop' lookup — even a simple version of it."""
    q = query.lower()
    query_terms = set(re.findall(r"[a-z0-9_]{3,}", q))
    matches = []

    for t in triples:
        subject = t["subject"].lower()
        obj = t["object"].lower()
        entity_terms = set(re.findall(r"[a-z0-9_]{3,}", f"{subject} {obj}"))
        if subject in q or obj in q or query_terms.intersection(entity_terms):
            matches.append(t)

    return matches


def format_graph_answer(triples, source_filter=None, limit=30):
    """Formats knowledge graph triples into a readable relationship list —
    used to answer 'what are the relationships/entities' type questions
    directly from the graph, without needing the LLM to guess."""
    if source_filter:
        triples = [t for t in triples if t["source"] == source_filter]

    if not triples:
        return "No relationships found in the knowledge graph for this."

    lines = [f"Found {len(triples)} relationship(s):\n"]
    for t in triples[:limit]:
        detail = f"; details: {t['details']}" if t.get("details") else ""
        lines.append(
            f"- {t['subject']}  --[{t['relation']}]-->  {t['object']} "
            f"(from {t['source']}{detail})"
        )

    if len(triples) > limit:
        lines.append(f"\n...and {len(triples) - limit} more.")

    return "\n".join(lines)