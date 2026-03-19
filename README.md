# askmydocs/README.md — Setup, architecture, API, CLI, and evaluation guide for AskMyDocs

## AskMyDocs

AskMyDocs is a fully local RAG system for querying private PDF documents with a defensive answer policy. The stack uses Ollama for generation, BGE embeddings for dense retrieval, ChromaDB for persistence, BM25 for sparse retrieval, an ensemble retriever for hybrid search, and a cross-encoder reranker for final context selection.

## Features

- Hybrid retrieval with BM25 and Chroma-backed dense search
- Cross-encoder reranking over the top 10 retrieved chunks
- Incremental ingestion with PDF hash detection and deleted-file cleanup
- Optional HyDE query rewriting and answer verification
- Session-scoped conversation memory with the last 5 turns
- FastAPI service with ingestion, query, health, and document management endpoints
- Structured Loguru logging to [`logs/askmydocs.log`](/Users/hemasai/Documents/langchainproj/askmydocs/logs/askmydocs.log)
- Offline RAGAS evaluation harness
- Pytest suite covering ingestion, retrieval, RAG behavior, and API endpoints

## Project Layout

```text
askmydocs/
├── app/
│   ├── api/
│   ├── core/
│   ├── ingest/
│   ├── rag/
│   ├── retriever/
│   └── vectorstore/
├── data/
│   ├── documents/
│   └── eval/
├── logs/
├── scripts/
├── tests/
├── .env.example
├── main.py
└── requirements.txt
```

## Setup

```bash
cd askmydocs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull mistral
```

The system is local-only. No cloud APIs are used.

## Ingestion

Place PDFs in [`data/documents`](/Users/hemasai/Documents/langchainproj/askmydocs/data/documents) and then either start the API or run the CLI. Incremental ingestion computes an MD5 hash for each file, skips unchanged PDFs, updates changed PDFs, removes deleted PDFs from Chroma, and rebuilds the persisted BM25 index only when document state changes.

## CLI Usage

```bash
python main.py
```

The CLI initializes the defensive RAG engine, keeps conversation memory for the current process, and answers questions against the indexed documents.

## API Usage

Start the server:

```bash
uvicorn app.api.main:app --reload --port 8000
```

Available endpoints:

- `POST /ingest`
- `POST /ask`
- `GET /documents`
- `DELETE /documents/{filename}`
- `GET /health`

Example ask request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the goals of artificial intelligence?"}'
```

## Evaluation

Sample evaluation pairs live in [`data/eval/qa_pairs.json`](/Users/hemasai/Documents/langchainproj/askmydocs/data/eval/qa_pairs.json). Add your own entries using this format:

```json
[
  {
    "question": "What is the document saying?",
    "ground_truth": "A short reference answer grounded in the documents."
  }
]
```

Run the evaluation harness:

```bash
python scripts/evaluate.py
```

The script writes a JSON report to [`logs/ragas_report.json`](/Users/hemasai/Documents/langchainproj/askmydocs/logs/ragas_report.json). If Ollama is not running, the script exits cleanly and records a skipped report instead of crashing.

## Testing

```bash
pytest tests/ -v
```

The tests mock heavy model dependencies so the suite can run without downloading embedding or reranker models.

## Configuration

All runtime settings are loaded from `.env` via [`app/core/config.py`](/Users/hemasai/Documents/langchainproj/askmydocs/app/core/config.py). No module should hard-code retrieval thresholds, model names, or storage paths.
