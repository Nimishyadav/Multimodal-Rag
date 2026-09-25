import os
import re
import sys
import argparse
import difflib

from src.excel_reader import workbook_relationship_triples
from src.knowledge_graph import load_graph, save_graph
from src.pyvis_graph import render_graph

triples = load_graph()
source_files = [
    filename
    for filename in sorted(os.listdir("data"))
    if filename.lower().endswith(
        (".xlsx", ".xls", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")
    )
    and not filename.startswith("~$")
]
workbook_paths = [
    os.path.join("data", filename)
    for filename in sorted(os.listdir("data"))
    if filename.lower().endswith((".xlsx", ".xls"))
    and not filename.startswith("~$")
]
excel_triples = workbook_relationship_triples(workbook_paths)
triples.extend(excel_triples)
unique_triples = []
seen_triples = set()
for triple in triples:
    triple_key = tuple(sorted(triple.items()))
    if triple_key not in seen_triples:
        seen_triples.add(triple_key)
        unique_triples.append(triple)
triples = unique_triples
save_graph(triples, source_files=source_files)
print("Graph files refreshed in storage/.")

parser = argparse.ArgumentParser(description="Visualize the multimodal knowledge graph.")
parser.add_argument(
    "source",
    nargs="?",
    help="Optional source or entity filter, such as 'sample pdf.pdf'.",
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument(
    "--excel",
    action="store_true",
    help="Show only relationships between the Excel workbooks.",
)
mode.add_argument(
    "--workbooks",
    nargs="+",
    metavar="FILE",
    help="Show relationships for one or more named Excel workbooks.",
)
mode.add_argument(
    "--image",
    action="store_true",
    help="Show only relationships from image files.",
)
args = parser.parse_args()

if args.excel:
    workbook_names = {os.path.basename(path) for path in workbook_paths}
    triples = [
        triple for triple in triples
        if triple in excel_triples
        or triple.get("source") in workbook_names
        or triple.get("subject") in workbook_names
        or triple.get("object") in workbook_names
    ]
    print("Showing Excel workbook relationships only.")

if args.workbooks:
    selected_workbooks = {os.path.basename(name).lower() for name in args.workbooks}
    triples = [
        triple for triple in triples
        if triple.get("source", "").lower() in selected_workbooks
        or triple.get("subject", "").lower() in selected_workbooks
        or triple.get("object", "").lower() in selected_workbooks
    ]
    print("Showing selected Excel workbooks: " + ", ".join(args.workbooks))

if args.image:
    image_names = {
        filename.lower()
        for filename in os.listdir("data")
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    triples = [
        triple for triple in triples
        if triple.get("source", "").lower() in image_names
        or triple.get("subject", "").lower() in image_names
        or triple.get("object", "").lower() in image_names
        or triple.get("source") == "cross-image"
    ]
    print("Showing image relationships only.")

if args.source:
    source_filter = args.source
    source_names = sorted({
        triple.get("source", "")
        for triple in triples
        if triple.get("source")
    } | set(source_files))
    if source_filter not in source_names:
        close_matches = difflib.get_close_matches(
            source_filter,
            source_names,
            n=1,
            cutoff=0.75,
        )
        if close_matches:
            source_filter = close_matches[0]
            print(f"Source not found; using closest match: {source_filter}")
    triples = [
        t for t in triples
        if t["source"] == source_filter
        or t["subject"] == source_filter
        or t["object"] == source_filter
    ]
    print(f"Showing only relationships from: {source_filter}")

view_name = "all"
if args.excel:
    view_name = "excel"
elif args.workbooks:
    selected_name = "_".join(args.workbooks)
    view_name = "excel_" + (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", selected_name).strip("_")
        or "selected"
    )
elif args.image:
    view_name = "image"
elif args.source:
    view_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.source).strip("_") or "source"

view_sources = sorted({triple.get("source", "") for triple in triples if triple.get("source")})
save_graph(
    triples,
    path=os.path.join("storage", f"graph_{view_name}_view.pkl"),
    source_files=view_sources,
)
print(f"Readable graph view saved to storage/graph_{view_name}_view_readable.txt")

if not triples:
    print("No relationships found to visualize.")
    sys.exit()

print(f"Building interactive graph from {len(triples)} relationships...")
graph_path = os.path.join("storage", f"graph_{view_name}.html")
render_graph(triples, graph_path)
print(f"Interactive PyVis graph saved to {graph_path}")