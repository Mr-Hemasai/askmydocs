# askmydocs/app/retriever/hybrid.py — Hybrid BM25 and Chroma retrieval utilities
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever

from app.core.config import settings
from app.core.logger import get_logger
from app.vectorstore.embeddings import format_query_for_retrieval
from app.vectorstore.store import get_vectorstore, ingest_documents, load_bm25_payload

logger = get_logger()


class PrefixedRetriever(BaseRetriever):
    """Retriever wrapper that prefixes queries for dense BGE search."""

    base_retriever: Any

    class Config:
        """Allow arbitrary wrapped retriever instances."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Return documents for a BGE-prefixed query."""

        return self.base_retriever.invoke(format_query_for_retrieval(query))


class EmptyRetriever(BaseRetriever):
    """Retriever that returns no documents."""

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Return an empty result set."""

        return []


@dataclass
class RetrievalResult:
    """Structured hybrid retrieval output."""

    documents: list[Document]
    distance_map: dict[str, float]
    relevance_map: dict[str, float]


class HybridRetriever:
    """Hybrid retriever that combines BM25 and Chroma search."""

    def __init__(self) -> None:
        """Initialize persistent BM25 and dense retrievers."""

        ingest_documents()
        self.vectorstore = get_vectorstore()
        self.ensemble = self._build_ensemble()

    def _build_ensemble(self) -> EnsembleRetriever:
        """Build the weighted ensemble retriever."""

        payload = load_bm25_payload()
        documents: list[Document] = payload.get("documents", [])

        if documents:
            bm25_retriever: BaseRetriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = settings.TOP_K_RETRIEVAL
        else:
            bm25_retriever = EmptyRetriever()

        dense_retriever = PrefixedRetriever(
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K_RETRIEVAL})
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[settings.BM25_WEIGHT, settings.FAISS_WEIGHT],
        )

    def refresh(self) -> None:
        """Refresh the ensemble after index changes."""

        ingest_documents()
        self.vectorstore = get_vectorstore()
        self.ensemble = self._build_ensemble()

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve hybrid documents and dense score metadata."""

        documents = self.ensemble.invoke(query)
        distance_results = self.vectorstore.similarity_search_with_score(
            format_query_for_retrieval(query),
            k=settings.TOP_K_RETRIEVAL,
        )

        distance_map: dict[str, float] = {}
        relevance_map: dict[str, float] = {}

        for document, distance in distance_results:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            distance_map[chunk_id] = float(distance)
            relevance_map[chunk_id] = max(0.0, 1.0 / (1.0 + float(distance)))
            preview = document.page_content[:100].replace("\n", " ")
            logger.debug(f"Retrieved chunk preview: {preview}")
            logger.debug(f"Distance score for {chunk_id}: {distance}")

        return RetrievalResult(documents=documents, distance_map=distance_map, relevance_map=relevance_map)
