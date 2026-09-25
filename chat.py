import json
import os
import sys
from datetime import datetime

from src.retriever import (
    search, get_summary_chunks, is_summary_query,
    is_stats_query, get_stats_answer,
    is_graph_query,
    get_known_sources, find_mentioned_source,
    build_grounded_context,
    is_excel_relationship_query, get_excel_chunks,
    is_excel_query, asks_for_all_excel_files, build_excel_relationship_context,
)
from src.llm import generate_answer, generate_summary, classify_intent
from src.knowledge_graph import load_graph, format_graph_answer, query_graph
from src.excel_reader import workbook_relationship_triples


def save_answer_record(query, answer, sources):
    """Persist every answer so retrieved data can be reviewed later."""
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
    os.makedirs("storage", exist_ok=True)

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

if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:]).strip()
else:
    query = input("Ask a question: ").strip()

# Excel intent is resolved first because spreadsheet evidence is loaded directly
# from the current data files rather than inferred from the general index.
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

if intent == "STATS":
    answer = get_stats_answer()
    save_answer_record(query, answer, [])
    print("\nAnswer:\n")
    print(answer)

elif intent == "GRAPH":
    sources = get_known_sources()
    target_source = find_mentioned_source(query, sources)  # None = across all documents

    triples = load_graph()
    current_workbooks = [
        os.path.join("data", filename)
        for filename in sorted(os.listdir("data"))
        if filename.lower().endswith((".xlsx", ".xls"))
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
    if target_source and not is_overview_query:
        matching_triples = [
            triple for triple in matching_triples
            if triple["source"] == target_source
        ]
    if is_overview_query:
        answer = format_graph_answer(triples, limit=100)
    else:
        if is_excel_relationship_query(query):
            excel_chunks = get_excel_chunks()
            relation_context = build_excel_relationship_context(excel_chunks)
            relation_context += "\n\n" + build_grounded_context(query, excel_chunks)
        else:
            relation_context = build_grounded_context(query, search(query, top_k_per_source=6))
        if matching_triples:
            relation_context += "\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n" + "\n".join(
                f"{triple['subject']} --[{triple['relation']}]--> {triple['object']} "
                f"(source: {triple['source']})"
                for triple in matching_triples[:30]
            )
        if is_excel_relationship_query(query):
            relationship_report = relation_context.split(
                "\n\nEXCEL WORKBOOK EVIDENCE:", 1
            )[0]
            answer = relationship_report + "\n\nDetailed interpretation:\n" + generate_answer(
                relation_context, query
            )
        else:
            answer = generate_answer(relation_context, query)
        if not matching_triples:
            answer += "\n\n" + format_graph_answer(triples, source_filter=target_source)

    print("\nAnswer:\n")
    save_answer_record(query, answer, [triple.get("source", "") for triple in triples])
    print(answer)

elif intent == "SUMMARY":
    if is_excel_query(query):
        excel_chunks = get_excel_chunks()
        workbook_sources = sorted(set(chunk["source"] for chunk in excel_chunks))
        target_source = (
            None if asks_for_all_excel_files(query)
            else find_mentioned_source(query, workbook_sources)
        )
        results = [
            chunk for chunk in excel_chunks
            if not target_source or chunk["source"] == target_source
        ]
    else:
        sources = get_known_sources()
        target_source = find_mentioned_source(query, sources)
        results = get_summary_chunks(per_source_limit=8, source_filter=target_source)
    context = build_grounded_context(query, results)
    answer = generate_summary(context)

    print("\nAnswer:\n")
    save_answer_record(query, answer, [result["source"] for result in results])
    print(answer)

    print("\nSources used:")
    for s in set(r["source"] for r in results):
        print(f"- {s}")

else:  # NORMAL
    results = search(query, top_k_per_source=4)
    context = build_grounded_context(query, results)
    answer = generate_answer(context, query)

    print("\nAnswer:\n")
    save_answer_record(query, answer, [result["source"] for result in results])
    print(answer)

    print("\nSources used:")
    for source in sorted(set(r["source"] for r in results)):
        print(f"- {source}")