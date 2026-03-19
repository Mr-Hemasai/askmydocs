# askmydocs/app/rag/verifier.py — Optional answer faithfulness verification using a second LLM pass
from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class AnswerVerifier:
    """Verify whether an answer is supported by retrieved context."""

    def __init__(self, llm: ChatOllama | None = None) -> None:
        """Initialize the verifier."""

        self.enabled = settings.ENABLE_VERIFICATION
        self.llm = llm or ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)

    def verify(self, context: str, answer: str) -> bool:
        """Return whether the answer is fully supported by the context."""

        if not self.enabled:
            return True

        prompt = (
            "Given this context:\n"
            f"{context}\n\n"
            "And this answer:\n"
            f"{answer}\n\n"
            "Is this answer fully supported by the context?\n"
            "Reply with only: SUPPORTED or UNSUPPORTED"
        )
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error(f"LLM timeout during answer verification: {exc}")
            return False

        verdict = response.content.strip().upper()
        return verdict == "SUPPORTED"
