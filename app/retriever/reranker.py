# askmydocs/app/retriever/reranker.py — Cross-encoder reranking wrapper for retrieved chunks
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Return the shared cross-encoder reranker."""

    try:
        return CrossEncoder(settings.RERANKER_MODEL)
    except Exception as exc:
        logger.error(f"Embedding failure while loading reranker {settings.RERANKER_MODEL}: {exc}")
        raise


class RerankingRetriever:
    """Rerank retrieved documents with a cross-encoder."""

    def __init__(self) -> None:
        """Initialize the reranker."""

        self.model = get_reranker_model()

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank candidate documents and keep the top configured subset."""

        if not documents:
            return []

        limited_documents = documents[: settings.TOP_K_RETRIEVAL]
        pairs = [(query, document.page_content) for document in limited_documents]
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:
            logger.error(f"Embedding failure while reranking documents: {exc}")
            raise

        ranked = sorted(
            zip(limited_documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked_documents: list[Document] = []
        for document, score in ranked[: settings.TOP_K_RERANKED]:
            if float(score) < 0.0:
                logger.debug(f"Dropping chunk with negative rerank score: {score:.3f}")
                continue
            document.metadata["rerank_score"] = float(score)
            reranked_documents.append(document)

        if not reranked_documents:
            logger.warning("All reranked chunks had negative scores — returning refusal")
        return reranked_documents
