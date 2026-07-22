# askmydocs/app/rag/chain.py — Defensive RAG engine with hybrid retrieval, reranking, memory, and verification
from __future__ import annotations

from collections import deque
from typing import Any

from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.prompt import ANSWER_PROMPT
from app.rag.query_rewriter import QueryRewriter
from app.rag.verifier import AnswerVerifier
from app.retriever.hybrid import HybridRetriever
from app.retriever.reranker import RerankingRetriever

logger = get_logger()
REFUSAL_MESSAGE = "I could not find relevant information in the provided documents."


class WindowedMemory:
    """Keep the last ``k`` conversation turns as a formatted history string."""

    def __init__(self, k: int) -> None:
        """Initialize the sliding-window memory."""

        self._turns: deque[tuple[str, str]] = deque(maxlen=k)

    def load_history(self) -> str:
        """Return the retained turns formatted for prompting."""

        return "\n".join(f"Human: {question}\nAI: {answer}" for question, answer in self._turns)

    def save(self, question: str, answer: str) -> None:
        """Append a completed turn to the window."""

        self._turns.append((question, answer))

    def clear(self) -> None:
        """Drop all retained turns."""

        self._turns.clear()


class DefensiveRAG:
    """Production defensive RAG engine for local document question answering."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: RerankingRetriever | None = None,
        llm: ChatOllama | None = None,
        query_rewriter: QueryRewriter | None = None,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        """Initialize the RAG components."""

        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or RerankingRetriever()
        self.llm = llm or ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
        self.query_rewriter = query_rewriter or QueryRewriter(self.llm)
        self.verifier = verifier or AnswerVerifier(self.llm)
        self.memory = WindowedMemory(k=5)
        self.turn_number = 0

    def clear_memory(self) -> None:
        """Clear conversation history for the current session."""

        self.memory.clear()
        self.turn_number = 0

    def validate_query(self, query: str) -> tuple[bool, str | None]:
        """Validate a user query before retrieval."""

        normalized = query.strip().lower()
        if len(normalized) < 3:
            return False, "Please ask a meaningful question related to the indexed documents."
        if normalized in {"hi", "hello", "hey", "hii"}:
            return False, "Hello. Ask a question about the indexed documents."
        return True, None

    def build_context(self, documents: list[Any]) -> str:
        """Build a context string from retrieved documents."""

        return "\n\n".join(document.page_content for document in documents)

    def _score_confidence(
        self,
        documents: list[Any],
        distance_map: dict[str, float],
        relevance_map: dict[str, float],
    ) -> tuple[str, bool]:
        """Compute a confidence label and whether the answer should be refused.

        Distances are only available for chunks that surfaced in dense
        retrieval, so a chunk retrieved solely via BM25 has no distance. Refusal
        is gated on the best available distance and is skipped entirely when no
        retrieved chunk carries one, avoiding false refusals of keyword hits.
        """

        distances = [
            distance_map[chunk_id]
            for document in documents
            if (chunk_id := str(document.metadata.get("chunk_id", ""))) in distance_map
        ]
        relevances = [
            relevance_map[chunk_id]
            for document in documents
            if (chunk_id := str(document.metadata.get("chunk_id", ""))) in relevance_map
        ]
        best_distance = min(distances) if distances else None
        best_relevance = max(relevances) if relevances else 0.0

        if best_distance is not None and best_distance > settings.DISTANCE_THRESHOLD:
            logger.warning(f"Distance threshold not met: {best_distance}")
            return "Low", True

        if best_relevance >= settings.CONFIDENCE_HIGH:
            return "High", False
        if best_relevance >= settings.CONFIDENCE_MEDIUM:
            return "Medium", False
        logger.warning(f"Low confidence answer with relevance {best_relevance}")
        return "Low", False

    def _generate_answer(self, question: str, context: str, history: str) -> str:
        """Generate an answer from the LLM using the strict prompt."""

        prompt = ANSWER_PROMPT.format(history=history or "No previous conversation.", context=context, question=question)
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error(f"LLM timeout while generating answer: {exc}")
            raise
        return response.content.strip()

    def ask(self, query: str) -> dict[str, Any]:
        """Answer a user question using defensive hybrid RAG."""

        logger.info(f"Query received: {query}")
        is_valid, validation_message = self.validate_query(query)
        if not is_valid:
            return {
                "answer": validation_message,
                "confidence": "None",
                "sources": [],
                "contexts": [],
                "turn_number": self.turn_number,
                "verified": False,
            }

        history = self.memory.load_history()
        rewritten_query = self.query_rewriter.rewrite(query, history=history)
        retrieval = self.retriever.retrieve(rewritten_query)
        reranked_documents = self.reranker.rerank(rewritten_query, retrieval.documents)

        if not reranked_documents:
            logger.warning("No reranked documents available for answer generation")
            return {
                "answer": REFUSAL_MESSAGE,
                "confidence": "Low",
                "sources": [],
                "contexts": [],
                "turn_number": self.turn_number,
                "verified": False,
            }

        confidence, should_refuse = self._score_confidence(
            reranked_documents,
            retrieval.distance_map,
            retrieval.relevance_map,
        )
        if should_refuse:
            return {
                "answer": REFUSAL_MESSAGE,
                "confidence": "Low",
                "sources": [],
                "contexts": [],
                "turn_number": self.turn_number,
                "verified": False,
            }

        context = self.build_context(reranked_documents)
        answer = self._generate_answer(query, context, history)
        verified = self.verifier.verify(context, answer)

        if not verified:
            answer = REFUSAL_MESSAGE
            confidence = "Unverified"

        self.turn_number += 1
        self.memory.save(query, answer)

        sources = [
            {
                "file": document.metadata.get("file_name", document.metadata.get("source")),
                "page": document.metadata.get("page_number", document.metadata.get("page")),
                "chunk_id": document.metadata.get("chunk_id"),
                "rerank_score": document.metadata.get("rerank_score"),
            }
            for document in reranked_documents
        ]
        logger.info(f"Answer generated with confidence {confidence}")
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "contexts": [document.page_content for document in reranked_documents],
            "turn_number": self.turn_number,
            "verified": verified,
        }
