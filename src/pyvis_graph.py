from collections import defaultdict
import os

from pyvis.network import Network


def render_graph(triples, output_path="storage/knowledge_graph.html"):
    """Render graph triples to an interactive PyVis HTML document."""
    network = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
        cdn_resources="in_line",
    )
    nodes = {}
    edges = set()
    edge_records = []
    source_by_node = defaultdict(set)

    for triple in triples:
        subject = str(triple.get("subject", ""))
        object_name = str(triple.get("object", ""))
        if not subject or not object_name:
            continue

        source = triple.get("source", "unknown source")
        relation = triple.get("relation", "related to")
        modality = triple.get("modality", "")
        details = triple.get("details", "")
        edge_key = (subject, object_name, relation, source, details)
        if edge_key in edges:
            continue
        edges.add(edge_key)

        source_by_node[subject].add(source)
        source_by_node[object_name].add(source)
        nodes.setdefault(subject, {"label": subject})
        nodes.setdefault(object_name, {"label": object_name})
        edge_title = f"Source: {source}"
        if modality:
            edge_title += f" | Modality: {modality}"
        if details:
            edge_title += f" | Details: {details}"
        edge_records.append((subject, object_name, relation, edge_title))

    for node, data in nodes.items():
        sources = ", ".join(sorted(source_by_node[node]))
        network.add_node(
            node,
            label=data["label"],
            title=f"Entity: {node}<br>Sources: {sources}",
            shape="dot",
            size=18,
        )

    for subject, object_name, relation, edge_title in edge_records:
        network.add_edge(
            subject,
            object_name,
            label=str(relation),
            title=edge_title,
            arrows="to",
        )

    network.set_options(
        """
        {
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 250},
            "barnesHut": {"gravitationalConstant": -3500, "centralGravity": 0.15, "springLength": 180, "springConstant": 0.04}
          },
          "edges": {"smooth": {"type": "dynamic"}, "font": {"size": 11, "align": "middle"}},
          "nodes": {"font": {"size": 14, "face": "Arial"}}
        }
        """
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    html = network.generate_html(name=os.path.basename(output_path), notebook=False)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)
    return output_path
