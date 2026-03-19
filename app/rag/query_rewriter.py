# askmydocs/app/rag/query_rewriter.py — Optional HyDE query rewriting with memory awareness
from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class QueryRewriter:
    """Optional HyDE query rewriter backed by Ollama."""

    def __init__(self, llm: ChatOllama | None = None) -> None:
        """Initialize the query rewriter."""

        self.enabled = settings.ENABLE_HYDE
        self.llm = llm or ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)

    def rewrite(self, query: str, history: str = "") -> str:
        """Rewrite a user query, resolving follow-ups using conversation history."""

        if history and history.strip() and history.strip() != "No previous conversation.":
            resolved = self._resolve_followup(query, history)
        else:
            resolved = query

        if not self.enabled:
            return resolved

        prompt = (
            "Write a short passage that would answer this question "
            f"if it appeared in a document: {resolved}"
        )
        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            return rewritten or resolved
        except Exception as exc:
            logger.error(f"LLM timeout during HyDE rewriting: {exc}")
            return resolved

    def _resolve_followup(self, query: str, history: str) -> str:
        """Use the LLM to rewrite a vague follow-up into a self-contained query."""

        prompt = (
            "Given this conversation history:\n"
            f"{history}\n\n"
            "Rewrite the follow-up question as a SHORT, keyword-dense search query "
            "(maximum 8 words). Use only the most important nouns and concepts. "
            "No verbs like 'provide' or 'describe'. No filler phrases. "
            "Return ONLY the rewritten query, nothing else.\n\n"
            f"Follow-up: {query}\n"
            "Rewritten query:"
        )
        try:
            response = self.llm.invoke(prompt)
            resolved = response.content.strip()
            resolved = resolved.strip('"').strip("'")
            logger.debug(f"Query resolved from '{query}' to '{resolved}'")
            return resolved or query
        except Exception as exc:
            logger.error(f"LLM timeout during follow-up resolution: {exc}")
            return query
