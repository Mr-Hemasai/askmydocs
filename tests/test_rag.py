# askmydocs/tests/test_rag.py — Tests for defensive RAG answers, confidence, memory, and verification
from __future__ import annotations

from langchain_core.documents import Document

from app.rag.chain import DefensiveRAG, REFUSAL_MESSAGE
from app.rag.verifier import AnswerVerifier
from app.retriever.hybrid import RetrievalResult


class FakeLLMResponse:
    """Simple LLM response container."""

    def __init__(self, content: str) -> None:
        """Store the model response content."""

        self.content = content


class FakeLLM:
    """Simple LLM stub."""

    def invoke(self, prompt: str) -> FakeLLMResponse:
        """Return a deterministic answer."""

        return FakeLLMResponse("Grounded answer.")


class FakeHybridRetriever:
    """Stub retriever returning a fixed retrieval result."""

    def __init__(self) -> None:
        """Initialize the retriever."""

        self.result = RetrievalResult(
            documents=[
                Document(page_content="Context one", metadata={"chunk_id": "c1", "file_name": "doc.pdf", "page_number": 1})
            ],
            distance_map={"c1": 0.3},
            relevance_map={"c1": 0.91},
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """Return the fixed retrieval result."""

        return self.result

    def refresh(self) -> None:
        """No-op refresh for tests."""


class FakeReranker:
    """Stub reranker that returns the input documents."""

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Return documents unchanged."""

        return documents


class FakeRewriter:
    """Stub query rewriter."""

    def rewrite(self, query: str, history: str = "") -> str:
        """Return the query unchanged."""

        return query


class FakeVerifier:
    """Stub answer verifier."""

    def __init__(self, verified: bool = True) -> None:
        """Store verification outcome."""

        self.verified = verified

    def verify(self, context: str, answer: str) -> bool:
        """Return the configured verification outcome."""

        return self.verified


def test_rag_ask_tracks_memory_and_turns() -> None:
    """Answer multiple questions and increment session turn count."""

    rag = DefensiveRAG(
        retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        query_rewriter=FakeRewriter(),
        verifier=FakeVerifier(True),
    )

    first = rag.ask("What is in the document?")
    second = rag.ask("And what else?")

    assert first["confidence"] == "High"
    assert first["verified"] is True
    assert first["turn_number"] == 1
    assert second["turn_number"] == 2

    rag.clear_memory()
    assert rag.turn_number == 0


def test_rag_verification_failure_replaces_answer() -> None:
    """Replace unsupported answers with the refusal message."""

    rag = DefensiveRAG(
        retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        query_rewriter=FakeRewriter(),
        verifier=FakeVerifier(False),
    )

    result = rag.ask("Unsupported?")
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["confidence"] == "Unverified"
    assert result["verified"] is False


def test_rag_rejects_invalid_query() -> None:
    """Reject greeting-only queries."""

    rag = DefensiveRAG(
        retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        query_rewriter=FakeRewriter(),
        verifier=FakeVerifier(True),
    )
    result = rag.ask("hi")
    assert result["confidence"] == "None"


class NoDistanceRetriever:
    """Retriever whose reranked docs carry no dense distance (BM25-only hit)."""

    def __init__(self) -> None:
        """Initialize the retriever with empty score maps."""

        self.result = RetrievalResult(
            documents=[
                Document(page_content="Keyword-only context", metadata={"chunk_id": "c9", "file_name": "doc.pdf", "page_number": 2})
            ],
            distance_map={},
            relevance_map={},
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """Return the fixed retrieval result."""

        return self.result


def test_rag_answers_bm25_only_hit_without_distance() -> None:
    """A chunk with no dense distance must not be force-refused."""

    rag = DefensiveRAG(
        retriever=NoDistanceRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        query_rewriter=FakeRewriter(),
        verifier=FakeVerifier(True),
    )

    result = rag.ask("What does the document say about keywords?")
    assert result["answer"] == "Grounded answer."
    assert result["confidence"] == "Low"
    assert result["contexts"] == ["Keyword-only context"]


class VerdictLLM:
    """LLM stub that returns a fixed verification verdict."""

    def __init__(self, verdict: str) -> None:
        """Store the verdict content to return."""

        self.verdict = verdict

    def invoke(self, prompt: str) -> FakeLLMResponse:
        """Return the configured verdict."""

        return FakeLLMResponse(self.verdict)


def test_verifier_accepts_supported_with_trailing_text() -> None:
    """A 'SUPPORTED.' verdict with punctuation is treated as supported."""

    verifier = AnswerVerifier(VerdictLLM("SUPPORTED."))
    verifier.enabled = True
    assert verifier.verify("context", "answer") is True


def test_verifier_rejects_unsupported() -> None:
    """An 'UNSUPPORTED' verdict is treated as not supported."""

    verifier = AnswerVerifier(VerdictLLM("UNSUPPORTED"))
    verifier.enabled = True
    assert verifier.verify("context", "answer") is False
