# askmydocs/app/ingest/loader.py — PDF loading utilities with stable metadata and file hashing
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


def compute_md5(file_path: Path) -> str:
    """Compute the MD5 checksum for a file."""

    try:
        digest = hashlib.md5()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        logger.error(f"File read error while hashing {file_path.name}: {exc}")
        raise


def build_file_metadata(file_path: Path, file_hash: str | None = None) -> dict[str, str | int]:
    """Build standard metadata for a PDF file."""

    resolved = file_path.resolve()
    stat_result = resolved.stat()
    return {
        "source": resolved.name,
        "source_path": str(resolved),
        "file_name": resolved.name,
        "file_hash": file_hash or compute_md5(resolved),
        "file_size": stat_result.st_size,
        "date_added": datetime.fromtimestamp(
            stat_result.st_ctime,
            tz=timezone.utc,
        ).isoformat(),
    }


def load_pdf(file_path: Path) -> list[Document]:
    """Load a single PDF into LangChain documents."""

    resolved = file_path.resolve()
    file_metadata = build_file_metadata(resolved)
    try:
        loader = PyPDFLoader(str(resolved))
        documents = loader.load()
    except Exception as exc:
        logger.error(f"File read error while loading {resolved.name}: {exc}")
        raise

    for page_number, document in enumerate(documents, start=1):
        document.metadata.update(file_metadata)
        document.metadata["page_number"] = page_number
    return documents


def load_documents(
    folder_path: Path | None = None,
    files: Iterable[Path] | None = None,
) -> list[Document]:
    """Load all requested PDF documents from disk."""

    target_dir = (folder_path or settings.documents_dir).resolve()
    pdf_files = list(files) if files is not None else sorted(target_dir.glob("*.pdf"))
    documents: list[Document] = []

    for pdf_file in pdf_files:
        if pdf_file.suffix.lower() != ".pdf":
            continue
        documents.extend(load_pdf(pdf_file))

    logger.info(f"Documents ingested from disk: {len(pdf_files)} files, {len(documents)} pages")
    return documents
