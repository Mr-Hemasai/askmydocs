# askmydocs/app/rag/chain.py — Defensive RAG engine with hybrid retrieval, reranking, memory, and verification
from __future__ import annotations

from typing import Any

from langchain.memory import ConversationBufferWindowMemory
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
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=False,
            memory_key="history",
            input_key="question",
            output_key="answer",
        )
        self.turn_number = 0

    def clear_memory(self) -> None:
        """Clear conversation history for the current session."""

        self.memory.clear()
        self.turn_number = 0

    def refresh_retriever(self) -> None:
        """Refresh indexes after ingestion changes."""

        self.retriever.refresh()

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
    ) -> tuple[str, float]:
        """Compute confidence from dense retrieval scores."""

        best_distance = float("inf")
        best_relevance = 0.0

        for document in documents:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id in distance_map:
                best_distance = min(best_distance, distance_map[chunk_id])
            if chunk_id in relevance_map:
                best_relevance = max(best_relevance, relevance_map[chunk_id])

        if best_distance > settings.DISTANCE_THRESHOLD:
            logger.warning(f"Distance threshold not met: {best_distance}")
            return "Low", best_distance

        if best_relevance >= settings.CONFIDENCE_HIGH:
            return "High", best_distance
        if best_relevance >= settings.CONFIDENCE_MEDIUM:
            return "Medium", best_distance
        logger.warning(f"Low confidence answer with relevance {best_relevance}")
        return "Low", best_distance

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
                "turn_number": self.turn_number,
                "verified": False,
            }

        history = str(self.memory.load_memory_variables({}).get("history", ""))
        rewritten_query = self.query_rewriter.rewrite(query, history=history)
        retrieval = self.retriever.retrieve(rewritten_query)
        reranked_documents = self.reranker.rerank(rewritten_query, retrieval.documents)

        if not reranked_documents:
            logger.warning("No reranked documents available for answer generation")
            return {
                "answer": REFUSAL_MESSAGE,
                "confidence": "Low",
                "sources": [],
                "turn_number": self.turn_number,
                "verified": False,
            }

        confidence, best_distance = self._score_confidence(
            reranked_documents,
            retrieval.distance_map,
            retrieval.relevance_map,
        )
        if best_distance > settings.DISTANCE_THRESHOLD:
            return {
                "answer": REFUSAL_MESSAGE,
                "confidence": "Low",
                "sources": [],
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
        self.memory.save_context({"question": query}, {"answer": answer})

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
            "turn_number": self.turn_number,
            "verified": verified,
        }
