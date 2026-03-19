# askmydocs/tests/test_retrieval.py — Tests for hybrid retrieval and reranking behavior
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.retriever import hybrid
from app.retriever.reranker import RerankingRetriever


class FakeRetriever:
    """Simple retriever that returns preloaded documents."""

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the retriever."""

        self.documents = documents

    def invoke(self, query: str, config: dict[str, Any] | None = None) -> list[Document]:
        """Return the stored documents."""

        return self.documents


class FakeVectorStore:
    """Minimal vector store stub for retrieval tests."""

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the fake vector store."""

        self.documents = documents

    def as_retriever(self, search_kwargs: dict[str, int]) -> FakeRetriever:
        """Return a retriever over stored documents."""

        return FakeRetriever(self.documents)

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return deterministic distance scores."""

        return [(document, 0.2 + index * 0.1) for index, document in enumerate(self.documents[:k])]


class FakeCrossEncoder:
    """Simple cross-encoder stub."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return descending scores based on position."""

        return [float(len(pairs) - index) for index, _ in enumerate(pairs)]


def test_hybrid_retrieval(monkeypatch: Any) -> None:
    """Combine BM25 and dense results into a retrieval bundle."""

    documents = [
        Document(page_content="alpha beta", metadata={"chunk_id": "c1"}),
        Document(page_content="gamma delta", metadata={"chunk_id": "c2"}),
    ]
    monkeypatch.setattr(hybrid, "ingest_documents", lambda: None)
    monkeypatch.setattr(hybrid, "get_vectorstore", lambda: FakeVectorStore(documents))
    monkeypatch.setattr(hybrid, "load_bm25_payload", lambda: {"documents": documents})

    retriever = hybrid.HybridRetriever()
    result = retriever.retrieve("alpha")

    assert len(result.documents) >= 2
    assert result.distance_map["c1"] == 0.2
    assert round(result.relevance_map["c1"], 3) == 0.833


def test_reranker(monkeypatch: Any) -> None:
    """Rerank documents with the configured top-k cutoff."""

    documents = [
        Document(page_content="doc one", metadata={"chunk_id": "c1"}),
        Document(page_content="doc two", metadata={"chunk_id": "c2"}),
    ]
    monkeypatch.setattr("app.retriever.reranker.get_reranker_model", lambda: FakeCrossEncoder())
    reranker = RerankingRetriever()
    reranked = reranker.rerank("question", documents)

    assert len(reranked) == 2
    assert reranked[0].metadata["rerank_score"] >= reranked[1].metadata["rerank_score"]
