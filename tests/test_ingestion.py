# askmydocs/tests/test_ingestion.py — Tests for PDF loading, chunking, and ingestion change detection
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingest.loader import compute_md5, load_documents
from app.ingest.splitter import split_documents
from app.vectorstore import store


class FakeVectorStore:
    """Minimal in-memory vector store for ingestion tests."""

    def __init__(self) -> None:
        """Initialize the fake vector store."""

        self.documents = {}

    def add_documents(self, documents: list[Any], ids: list[str]) -> None:
        """Store documents by id."""

        for document, doc_id in zip(documents, ids):
            self.documents[doc_id] = document

    def get(
        self,
        where: dict[str, str] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Return stored documents optionally filtered by source path."""

        items = list(self.documents.items())
        if where and "source_path" in where:
            items = [
                item
                for item in items
                if item[1].metadata.get("source_path") == where["source_path"]
            ]
        return {
            "ids": [doc_id for doc_id, _ in items],
            "documents": [document.page_content for _, document in items],
            "metadatas": [document.metadata for _, document in items],
        }

    def delete(self, ids: list[str]) -> None:
        """Delete stored documents."""

        for doc_id in ids:
            self.documents.pop(doc_id, None)


def test_pdf_loading_and_chunking(synthetic_pdf: Path) -> None:
    """Load a synthetic PDF and split it into chunks."""

    documents = load_documents(files=[synthetic_pdf])
    assert len(documents) >= 1
    assert "Artificial intelligence" in documents[0].page_content

    chunks = split_documents(documents)
    assert len(chunks) >= 1
    assert chunks[0].metadata["file_name"] == "synthetic.pdf"


def test_hash_detection(monkeypatch: Any, synthetic_pdf: Path) -> None:
    """Detect unchanged files and avoid rebuilding indexes."""

    fake_vectorstore = FakeVectorStore()
    monkeypatch.setattr(store, "get_vectorstore", lambda: fake_vectorstore)

    first_result = store.ingest_documents()
    assert first_result.files_added == ["synthetic.pdf"]
    assert first_result.documents_changed is True
    assert first_result.chunks_indexed >= 1

    second_result = store.ingest_documents()
    assert second_result.documents_changed is False
    assert second_result.chunks_indexed == 0

    original_hash = compute_md5(synthetic_pdf)
    synthetic_pdf.write_bytes(synthetic_pdf.read_bytes() + b"\n% update")
    assert compute_md5(synthetic_pdf) != original_hash

    third_result = store.ingest_documents()
    assert third_result.files_updated == ["synthetic.pdf"]
    assert third_result.documents_changed is True
