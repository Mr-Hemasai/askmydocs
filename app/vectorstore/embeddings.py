# askmydocs/app/vectorstore/embeddings.py — Embedding model helpers and BGE query formatting
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance."""

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.error(f"Embedding failure while loading model {settings.EMBEDDING_MODEL}: {exc}")
        raise


def format_query_for_retrieval(query: str) -> str:
    """Prefix a query for BGE retrieval without altering stored document text."""

    normalized = query.strip()
    if not normalized:
        return normalized
    return f"{BGE_QUERY_PREFIX}{normalized}"
