# askmydocs/app/vectorstore/store.py — Persistent Chroma and BM25 index management with change detection
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logger import get_logger
from app.ingest.loader import compute_md5, load_documents
from app.ingest.splitter import split_documents
from app.vectorstore.embeddings import get_embedding_model

logger = get_logger()


@dataclass
class IngestionResult:
    """Structured result from an ingestion run."""

    status: str
    files_added: list[str]
    files_updated: list[str]
    files_removed: list[str]
    chunks_indexed: int
    documents_changed: bool


def ensure_storage_dirs() -> None:
    """Ensure all required persistent directories exist."""

    settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the shared persistent Chroma client."""

    ensure_storage_dirs()
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def get_vectorstore() -> Chroma:
    """Return the Chroma vector store wrapper."""

    try:
        return Chroma(
            client=get_chroma_client(),
            collection_name="askmydocs",
            embedding_function=get_embedding_model(),
        )
    except Exception as exc:
        logger.error(f"Embedding failure while initializing Chroma: {exc}")
        raise


def load_doc_hashes() -> dict[str, str]:
    """Load persisted document hashes from disk."""

    if not settings.doc_hashes_path.exists():
        return {}
    try:
        return json.loads(settings.doc_hashes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"File read error while loading document hashes: {exc}")
        raise


def save_doc_hashes(doc_hashes: dict[str, str]) -> None:
    """Persist document hashes to disk."""

    settings.doc_hashes_path.write_text(
        json.dumps(doc_hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_document_manifest() -> list[dict[str, Any]]:
    """Load the indexed document manifest."""

    if not settings.document_manifest_path.exists():
        return []
    try:
        return json.loads(settings.document_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"File read error while loading document manifest: {exc}")
        raise


def save_document_manifest(entries: list[dict[str, Any]]) -> None:
    """Persist the indexed document manifest to disk."""

    settings.document_manifest_path.write_text(
        json.dumps(entries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_bm25_payload() -> dict[str, Any]:
    """Load the persisted BM25 payload from disk."""

    if not settings.bm25_index_path.exists():
        return {"documents": [], "tokenized_corpus": []}
    try:
        with settings.bm25_index_path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError) as exc:
        logger.error(f"File read error while loading BM25 index: {exc}")
        raise


def save_bm25_payload(documents: list[Document]) -> None:
    """Persist BM25 documents and tokens to disk."""

    tokenized_corpus = [document.page_content.lower().split() for document in documents]
    payload = {
        "documents": documents,
        "tokenized_corpus": tokenized_corpus,
        "bm25": BM25Okapi(tokenized_corpus) if tokenized_corpus else None,
    }
    with settings.bm25_index_path.open("wb") as handle:
        pickle.dump(payload, handle)


def delete_chunks_for_file(vectorstore: Chroma, file_path: Path) -> int:
    """Delete all chunks belonging to a file from Chroma."""

    resolved = str(file_path.resolve())
    collection_data = vectorstore.get(where={"source_path": resolved})
    chunk_ids = collection_data.get("ids", [])
    if chunk_ids:
        vectorstore.delete(ids=chunk_ids)
    return len(chunk_ids)


def rebuild_bm25_from_chroma(vectorstore: Chroma) -> None:
    """Rebuild the persisted BM25 index from all stored Chroma chunks."""

    collection_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []

    for page_content, metadata in zip(
        collection_data.get("documents", []),
        collection_data.get("metadatas", []),
    ):
        if not page_content or metadata is None:
            continue
        documents.append(Document(page_content=page_content, metadata=metadata))

    save_bm25_payload(documents)

    manifest_map: dict[str, dict[str, Any]] = {}
    for document in documents:
        file_name = str(document.metadata.get("file_name", "unknown"))
        entry = manifest_map.setdefault(
            file_name,
            {
                "filename": file_name,
                "hash": document.metadata.get("file_hash", ""),
                "chunk_count": 0,
                "file_size": int(document.metadata.get("file_size", 0)),
                "date_added": document.metadata.get("date_added", ""),
            },
        )
        entry["chunk_count"] += 1
    save_document_manifest(sorted(manifest_map.values(), key=lambda item: item["filename"]))


def _build_current_hash_map() -> dict[str, str]:
    """Compute the current hash map for all PDFs on disk."""

    current_hashes: dict[str, str] = {}
    for pdf_path in sorted(settings.documents_dir.glob("*.pdf")):
        current_hashes[pdf_path.name] = compute_md5(pdf_path)
    return current_hashes


def ingest_documents() -> IngestionResult:
    """Ingest changed documents into Chroma and rebuild BM25 when needed."""

    ensure_storage_dirs()
    vectorstore = get_vectorstore()
    previous_hashes = load_doc_hashes()
    current_hashes = _build_current_hash_map()

    removed_files = sorted(set(previous_hashes) - set(current_hashes))
    changed_files = sorted(
        file_name
        for file_name, file_hash in current_hashes.items()
        if previous_hashes.get(file_name) != file_hash
    )
    added_files = sorted(file_name for file_name in changed_files if file_name not in previous_hashes)
    updated_files = sorted(file_name for file_name in changed_files if file_name in previous_hashes)

    for file_name in removed_files:
        delete_chunks_for_file(vectorstore, settings.documents_dir / file_name)

    chunks_indexed = 0
    if changed_files:
        file_paths = [settings.documents_dir / file_name for file_name in changed_files]
        for file_path in file_paths:
            delete_chunks_for_file(vectorstore, file_path)

        documents = load_documents(files=file_paths)
        chunks = split_documents(documents)
        if chunks:
            ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            vectorstore.add_documents(documents=chunks, ids=ids)
            chunks_indexed = len(chunks)

    documents_changed = bool(changed_files or removed_files or not settings.bm25_index_path.exists())
    if documents_changed:
        rebuild_bm25_from_chroma(vectorstore)
        save_doc_hashes(current_hashes)

    result = IngestionResult(
        status="ok",
        files_added=added_files,
        files_updated=updated_files,
        files_removed=removed_files,
        chunks_indexed=chunks_indexed,
        documents_changed=documents_changed,
    )
    logger.info(f"Documents ingested: {asdict(result)}")
    return result


def delete_document(filename: str) -> IngestionResult:
    """Delete a document from disk and indexes."""

    target = settings.documents_dir / filename
    if not target.exists():
        raise FileNotFoundError(filename)

    try:
        target.unlink()
    except OSError as exc:
        logger.error(f"File read error while deleting {filename}: {exc}")
        raise

    return ingest_documents()


def get_indexed_documents() -> list[dict[str, Any]]:
    """Return the persisted document manifest."""

    return load_document_manifest()
