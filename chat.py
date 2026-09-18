from src.retriever import (
    search, is_summary_query, get_summary_chunks,
    is_stats_query, get_stats_answer,
    get_known_sources, find_mentioned_source
)
from src.llm import generate_answer, generate_summary

query = input("Ask a question: ")

if is_stats_query(query):
    answer = get_stats_answer()
    print("\nAnswer:\n")
    print(answer)

elif is_summary_query(query):
    sources = get_known_sources()
    target_source = find_mentioned_source(query, sources)  # None if no specific file named

    results = get_summary_chunks(per_source_limit=8, source_filter=target_source)
    context = "\n\n".join([r["text"] for r in results])
    answer = generate_summary(context)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used:")
    for s in set(r["source"] for r in results):
        print(f"- {s}")

else:
    results = search(query, top_k_per_source=2)
    context = "\n\n".join([r["text"] for r in results])
    answer = generate_answer(context, query)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used:")
    for s in set(r["source"] for r in results):
        print(f"- {s}")