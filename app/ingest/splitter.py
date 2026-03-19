# askmydocs/app/ingest/splitter.py — Document chunking utilities for retrieval
from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings


def split_documents(documents: list[Document]) -> list[Document]:
    """Split loaded documents into retrieval chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        source = chunk.metadata.get("source_path", "unknown")
        page = chunk.metadata.get("page_number", chunk.metadata.get("page", 0))
        chunk.metadata["chunk_id"] = f"{source}::page::{page}::chunk::{index}"
    return chunks
