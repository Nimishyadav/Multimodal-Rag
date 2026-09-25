import os
import pickle
import json
import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from src.pdf_reader import read_pdf
from src.image_reader import read_image_segments
from src.excel_reader import (
    read_excel, workbook_relationship_triples, excel_entity_triples
)
from src.chunker import chunk_segment
from src.embedder import create_embeddings
from src.vector_store import VectorStore
from src.knowledge_graph import (
    build_graph, save_graph, add_structural_triples, load_graph, query_graph,
    format_graph_answer
)
from src.retriever import (
    search, get_summary_chunks, is_summary_query, is_stats_query, get_stats_answer,
    is_graph_query, get_known_sources, find_mentioned_source, build_grounded_context,
    is_excel_relationship_query, get_excel_chunks, is_excel_query,
    asks_for_all_excel_files, build_excel_relationship_context,
)
from src.llm import generate_answer, generate_summary, classify_intent
from src.pyvis_graph import render_graph

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("storage", exist_ok=True)

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG — PDF / Image / Excel")


# =====================================================================
# Shared helpers (same logic as chat.py / ingest.py, wrapped as functions
# so Streamlit can call them without running a script from top to bottom)
# =====================================================================

def save_answer_record(query, answer, sources):
    """Same readable-history logic used by chat.py — every answer gets
    written to plain text/JSONL files under storage/, so anyone can open
    and read them without touching any code."""
    query_lower = query.lower()
    is_graph_record = any(
        term in query_lower
        for term in ("graph", "relationship", "relationships", "entity", "entities", "connected")
    )
    if any(term in query_lower for term in ("excel", "spreadsheet", "workbook", "worksheet", ".xlsx", ".xls")):
        category = "excel"
    elif any(term in query_lower for term in ("image", "picture", "photo", "diagram", ".png", ".jpg", ".jpeg")):
        category = "image"
    elif any(term in query_lower for term in ("pdf", "document")):
        category = "pdf"
    else:
        category = "mixed"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "query": query,
        "sources": sorted(set(sources)),
        "answer": answer,
    }

    with open("storage/answer_history.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open("storage/current_answer.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    history_text = (
        f"[{record['timestamp']}] CATEGORY: {category}\n"
        f"QUESTION: {query}\n"
        f"SOURCES: {', '.join(record['sources']) or 'none'}\n"
        f"ANSWER:\n{answer}\n\n{'=' * 80}\n\n"
    )

    with open("storage/all_files_answer_history.txt", "a", encoding="utf-8") as file:
        file.write(history_text)
    with open("storage/current_answer.txt", "w", encoding="utf-8") as file:
        file.write(history_text)
    with open(f"storage/{category}_answer_history.txt", "a", encoding="utf-8") as file:
        file.write(history_text)
    if is_graph_record:
        with open("storage/graph_answer_history.txt", "a", encoding="utf-8") as file:
            file.write(history_text)


def answer_question(query):
    """Exact same routing logic as chat.py (STATS / GRAPH / SUMMARY / NORMAL),
    just returning the values instead of printing them."""
    started_at = time.perf_counter()
    debug_chunks = []
    debug_triples = []

    if is_excel_query(query):
        intent = "GRAPH" if is_excel_relationship_query(query) else "SUMMARY"
    elif is_stats_query(query):
        intent = "STATS"
    elif is_graph_query(query):
        intent = "GRAPH"
    elif is_summary_query(query):
        intent = "SUMMARY"
    else:
        intent = classify_intent(query)

    sources_used = []

    if intent == "STATS":
        answer = get_stats_answer()

    elif intent == "GRAPH":
        sources = get_known_sources()
        target_source = find_mentioned_source(query, sources)
        triples = load_graph()
        current_workbooks = [
            os.path.join(DATA_FOLDER, f) for f in sorted(os.listdir(DATA_FOLDER))
            if f.lower().endswith((".xlsx", ".xls")) and not f.startswith("~$")
        ]
        triples += workbook_relationship_triples(current_workbooks)

        is_overview_query = any(
            phrase in query.lower()
            for phrase in (
                "show the knowledge graph", "show knowledge graph", "view the graph",
                "all relationships", "all entities", "graph of everything",
                "relationships in all files", "knowledge graph of all",
            )
        )
        matching_triples = query_graph(query, triples)
        debug_triples = matching_triples[:100] if matching_triples else triples[:100]
        if target_source and not is_overview_query:
            matching_triples = [t for t in matching_triples if t["source"] == target_source]

        if is_overview_query:
            answer = format_graph_answer(triples, limit=100)
        else:
            if is_excel_relationship_query(query):
                excel_chunks = get_excel_chunks()
                debug_chunks = excel_chunks
                relation_context = build_excel_relationship_context(excel_chunks)
                relation_context += "\n\n" + build_grounded_context(query, excel_chunks)
                sources_used = [c["source"] for c in excel_chunks]
            else:
                results = search(query, top_k_per_source=6)
                debug_chunks = results
                relation_context = build_grounded_context(query, results)
                sources_used = [r["source"] for r in results]

            if matching_triples:
                relation_context += "\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n" + "\n".join(
                    f"{t['subject']} --[{t['relation']}]--> {t['object']} (source: {t['source']})"
                    for t in matching_triples[:30]
                )

            if is_excel_relationship_query(query):
                relationship_report = relation_context.split("\n\nEXCEL WORKBOOK EVIDENCE:", 1)[0]
                answer = relationship_report + "\n\nDetailed interpretation:\n" + generate_answer(
                    relation_context, query
                )
            else:
                answer = generate_answer(relation_context, query)

            if not matching_triples:
                answer += "\n\n" + format_graph_answer(triples, source_filter=target_source)

        sources_used = sources_used or [t.get("source", "") for t in triples]

    elif intent == "SUMMARY":
        if is_excel_query(query):
            excel_chunks = get_excel_chunks()
            workbook_sources = sorted(set(c["source"] for c in excel_chunks))
            target_source = (
                None if asks_for_all_excel_files(query)
                else find_mentioned_source(query, workbook_sources)
            )
            results = [c for c in excel_chunks if not target_source or c["source"] == target_source]
        else:
            sources = get_known_sources()
            target_source = find_mentioned_source(query, sources)
            results = get_summary_chunks(per_source_limit=8, source_filter=target_source)
        context = build_grounded_context(query, results)
        debug_chunks = results
        answer = generate_summary(context)
        sources_used = [r["source"] for r in results]

    else:  # NORMAL
        results = search(query, top_k_per_source=4)
        debug_chunks = results
        context = build_grounded_context(query, results)
        answer = generate_answer(context, query)
        sources_used = [r["source"] for r in results]

    save_answer_record(query, answer, sources_used)
    debug = {
        "retrieval_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "chunks_used": len(debug_chunks),
        "retrieved_chunks": [
            {
                "score": chunk.get("retrieval_score", chunk.get("score")),
                "source_type": chunk.get("modality", chunk.get("type", "TEXT")),
                "filename": chunk.get("source", "unknown"),
                "page": chunk.get("page"),
                "sheet": chunk.get("sheet"),
                "rows": (
                    f"{chunk['row_start']}-{chunk.get('row_end', chunk['row_start'])}"
                    if chunk.get("row_start") is not None else None
                ),
                "text": chunk.get("text", ""),
            }
            for chunk in debug_chunks
        ],
        "graph_triples": [
            {
                "subject": triple.get("subject", ""),
                "relation": triple.get("relation", ""),
                "object": triple.get("object", ""),
                "source": triple.get("source", ""),
            }
            for triple in debug_triples
        ],
    }
    return answer, sorted(set(sources_used)), debug


def run_ingestion(progress_callback=None):
    """Exact same logic as ingest.py, wrapped as a callable function."""
    EXCEL_EXTENSIONS = (".xlsx", ".xls")
    PDF_EXTENSIONS = (".pdf",)
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

    all_files = sorted(os.listdir(DATA_FOLDER))
    input_files = [
        f for f in all_files
        if f.lower().endswith(EXCEL_EXTENSIONS + PDF_EXTENSIONS + IMAGE_EXTENSIONS)
        and not f.startswith("~$")
    ]
    if not input_files:
        return "No supported files found in data/ folder."

    all_chunks = []
    stats_by_source = {}
    sources_text = {}
    structural_counts = {}

    for filename in input_files:
        path = os.path.join(DATA_FOLDER, filename)
        extension = os.path.splitext(filename)[1].lower()
        if progress_callback:
            progress_callback(f"Reading {extension[1:].upper()}: {filename}...")

        if extension in EXCEL_EXTENSIONS:
            segments, stats = read_excel(path)
        elif extension in PDF_EXTENSIONS:
            segments, stats = read_pdf(path)
        else:
            segments = read_image_segments(path)
            stats = {"pages": 0, "tables": 0, "images": 1}

        stats_by_source[filename] = stats
        sources_text[filename] = "\n".join(s["text"] for s in segments)

        modality_counts = {"TABLE": 0, "IMAGE": 0, "TEXT": 0}
        for segment_index, seg in enumerate(segments):
            seg.setdefault("id", f"{filename}:segment:{segment_index}")
            seg.setdefault("modality", seg["type"])
            chunks = chunk_segment(seg, filename, segment_index)
            all_chunks.extend(chunks)
            modality = seg.get("modality", seg.get("type", "TEXT")).upper()
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        structural_counts[filename] = modality_counts

    if progress_callback:
        progress_callback(f"Creating embeddings for {len(all_chunks)} chunks...")

    texts_only = [c["text"] for c in all_chunks]
    if not texts_only:
        return "No readable content was found in data/."

    embeddings = create_embeddings(texts_only)
    store = VectorStore(len(embeddings[0]))
    store.add(embeddings)
    store.save("storage/index.faiss")

    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open("storage/stats.pkl", "wb") as f:
        pickle.dump(stats_by_source, f)
    with open("storage/source_text.pkl", "wb") as f:
        pickle.dump(sources_text, f)

    if progress_callback:
        progress_callback("Building knowledge graph (this uses the LLM, may take a moment)...")

    triples = build_graph(sources_text, all_chunks)
    triples += add_structural_triples(structural_counts)

    excel_paths = [
        os.path.join(DATA_FOLDER, f) for f in input_files
        if f.lower().endswith(EXCEL_EXTENSIONS)
    ]
    triples += workbook_relationship_triples(excel_paths)
    triples += excel_entity_triples(excel_paths)

    save_graph(
        triples,
        source_files=input_files,
        chunks=all_chunks,
        stats_by_source=stats_by_source,
        source_text_by_source=sources_text,
    )

    return f"Done — {len(input_files)} file(s), {len(all_chunks)} chunk(s), {len(triples)} relationship(s)."


# =====================================================================
# SIDEBAR — upload + ingest
# =====================================================================
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Images, or Excel files",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            with open(os.path.join(DATA_FOLDER, uploaded.name), "wb") as f:
                f.write(uploaded.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to data/")

    st.divider()
    st.subheader("Files in data/")
    current_files = [f for f in sorted(os.listdir(DATA_FOLDER)) if not f.startswith("~$")]
    if current_files:
        for f in current_files:
            st.write(f"- {f}")
    else:
        st.write("_No files yet_")

    st.divider()
    if st.button("Process Files (Ingest)", type="primary", use_container_width=True):
        status_box = st.empty()
        with st.spinner("Processing..."):
            result = run_ingestion(progress_callback=lambda msg: status_box.info(msg))
        st.success(result)


# =====================================================================
# MAIN — tabs
# =====================================================================
tab_chat, tab_graph, tab_report, tab_history = st.tabs(
    ["Chat", "Knowledge Graph", "Readable Report", "Answer History"]
)

# ---------------- Chat tab ----------------
with tab_chat:
    st.subheader("Ask a question")

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")

    if submitted and query.strip():
        if not os.path.exists("storage/index.faiss"):
            st.warning("No index found yet. Upload files and click 'Process Files (Ingest)' first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources_used, debug = answer_question(query.strip())
                except Exception as e:
                    answer, sources_used, debug = f"Error: {e}", [], None
            st.session_state.chat_log.insert(0, (query.strip(), answer, sources_used, debug))

    for entry in st.session_state.chat_log:
        q, a, s = entry[:3]
        debug = entry[3] if len(entry) > 3 else None
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if s:
                st.caption("Sources: " + ", ".join(s))
            if debug:
                with st.expander("Retrieval debug details"):
                    metric_one, metric_two = st.columns(2)
                    metric_one.metric("Retrieval time", f"{debug['retrieval_time_ms']} ms")
                    metric_two.metric("Chunks used", debug["chunks_used"])
                    st.markdown("**Retrieved chunks**")
                    if debug["retrieved_chunks"]:
                        st.dataframe(debug["retrieved_chunks"], use_container_width=True)
                    else:
                        st.caption("No retrieved chunks for this response.")
                    st.markdown("**Graph triples**")
                    if debug["graph_triples"]:
                        st.dataframe(debug["graph_triples"], use_container_width=True)
                    else:
                        st.caption("No matching graph triples.")

# ---------------- Knowledge Graph tab ----------------
with tab_graph:
    st.subheader("Knowledge Graph Visualization")

    if not os.path.exists("storage/graph.pkl"):
        st.info("No graph yet. Upload files and click 'Process Files (Ingest)' first.")
    else:
        triples = load_graph()
        workbook_paths = [
            os.path.join(DATA_FOLDER, f) for f in sorted(os.listdir(DATA_FOLDER))
            if f.lower().endswith((".xlsx", ".xls")) and not f.startswith("~$")
        ]
        triples = list(triples) + workbook_relationship_triples(workbook_paths)

        all_sources = sorted(
            set(t.get("source", "") for t in triples if t.get("source"))
            | set(
                filename for filename in os.listdir(DATA_FOLDER)
                if not filename.startswith("~$")
            )
        )
        filter_choice = st.selectbox("Filter graph by source (optional)", ["All sources"] + all_sources)

        filtered = triples
        if filter_choice != "All sources":
            filtered = [
                t for t in triples
                if t.get("source") == filter_choice
                or t.get("subject") == filter_choice
                or t.get("object") == filter_choice
            ]

        st.write(f"Showing {len(filtered)} of {len(triples)} relationships")
        filtered_sources = sorted({
            triple.get("source", "")
            for triple in filtered
            if triple.get("source")
        })
        save_graph(
            filtered,
            path="storage/streamlit_selected_graph.pkl",
            source_files=filtered_sources or all_sources,
        )
        selected_report_path = "storage/streamlit_selected_graph_readable.txt"
        if os.path.exists(selected_report_path):
            with open(selected_report_path, "r", encoding="utf-8") as source_file:
                selected_report = source_file.read()
            with open("storage/graph_readable.txt", "w", encoding="utf-8") as report_file:
                report_file.write(selected_report)
        st.caption("The readable report is refreshed for the selected graph view.")

        if filtered:
            graph_path = render_graph(filtered, "storage/streamlit_graph.html")
            with open(graph_path, "r", encoding="utf-8") as file:
                components.html(file.read(), height=820, scrolling=True)
        else:
            st.write("No relationships to show for this filter.")

        st.divider()
        st.subheader("File Details")
        detail_choice = st.selectbox(
            "Select a file to read its complete extracted data",
            ["Select a file"] + all_sources,
        )
        if detail_choice != "Select a file":
            stats = {}
            source_text = ""
            try:
                with open("storage/stats.pkl", "rb") as file:
                    stats = pickle.load(file).get(detail_choice, {})
            except (FileNotFoundError, OSError, pickle.PickleError):
                pass
            try:
                with open("storage/source_text.pkl", "rb") as file:
                    source_text = pickle.load(file).get(detail_choice, "")
            except (FileNotFoundError, OSError, pickle.PickleError):
                pass

            if stats:
                st.json(stats)
            if source_text:
                st.text_area(
                    "Complete extracted file data",
                    source_text,
                    height=500,
                    key=f"details_{detail_choice}",
                )
            else:
                st.info("No extracted text is available. Run Process Files (Ingest) after adding the file.")

# ---------------- Readable report tab ----------------
with tab_report:
    st.subheader("Readable Knowledge Graph Report")
    st.caption("This report updates whenever the graph source filter changes.")
    report_path = "storage/streamlit_selected_graph_readable.txt"
    if not os.path.exists(report_path):
        report_path = "storage/graph_readable.txt"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button("Download report", content, file_name="selected_graph_readable.txt")
        st.text_area("Report contents", content, height=600)
    else:
        st.info("No report yet. Run 'Process Files (Ingest)' first.")

# ---------------- Answer history tab ----------------
with tab_history:
    st.subheader("Answer History")
    st.caption("This shows only the latest question and answer. The full archive remains in storage/all_files_answer_history.txt.")
    history_path = "storage/current_answer.txt"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button("Download history", content, file_name="answer_history.txt")
        st.text_area("History contents", content, height=600)
    else:
        st.info("No answers saved yet.")
