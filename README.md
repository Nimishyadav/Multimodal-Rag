# Multimodal GraphRAG

A Multimodal Retrieval-Augmented Generation (RAG) system that supports PDFs, Images, and Excel Workbooks. The system performs semantic retrieval using FAISS, generates Knowledge Graphs from extracted relationships, and provides interactive graph visualization using PyVis and Streamlit.

## Features

- PDF Question Answering
- Image OCR Support
- Excel Workbook Processing
- Multimodal Retrieval
- FAISS Vector Database
- Ollama Llama3 Integration
- Knowledge Graph Generation
- Entity Relationship Extraction
- Workbook Relationship Extraction
- Interactive Graph Visualization (PyVis)
- Streamlit Dashboard
- Retrieval Debug View
- Multi-document Retrieval

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama (Llama3)
- PyMuPDF
- Tesseract OCR
- OpenPyXL
- NetworkX
- PyVis

## Architecture

PDF / Images / Excel
        ↓
     Readers
        ↓
  Text Extraction
        ↓
     Chunking
        ↓
    Embeddings
        ↓
       FAISS

        +

 Relationship Extraction
        ↓
   Knowledge Graph
        ↓
      graph.pkl
        ↓
       PyVis

        +

    User Query
        ↓
     Retriever
        ↓
 Relevant Chunks
        ↓
      Llama3
        ↓
   Final Answer

## Project Structure

```text
src/
├── pdf_reader.py
├── image_reader.py
├── excel_reader.py
├── knowledge_graph.py
├── pyvis_graph.py
├── retriever.py
├── reranker.py
├── embedder.py
├── chunker.py
├── vector_store.py
├── llm.py

app.py
chat.py
ingest.py
visualize_graph.py
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Build Index

```bash
python ingest.py
```

### Run Chat Interface

```bash
python chat.py
```

### Run Streamlit Dashboard

```bash
streamlit run app.py
```

### Visualize Knowledge Graph

```bash
python visualize_graph.py
```

## Future Improvements

- Hybrid Search (BM25 + FAISS)
- Source Citations
- Chat Memory
- Graph-Aware Retrieval
- Advanced GraphRAG
- Multi-Agent Workflows