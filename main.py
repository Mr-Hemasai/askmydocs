# askmydocs/main.py — Command-line interface for local AskMyDocs querying
from __future__ import annotations

import sys

from app.core.logger import get_logger
from app.rag.chain import DefensiveRAG

logger = get_logger()


def write_line(message: str) -> None:
    """Write a line to stdout without using print."""

    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def main() -> int:
    """Run the interactive AskMyDocs CLI."""

    logger.info("Starting AskMyDocs CLI")
    try:
        rag = DefensiveRAG()
    except Exception as exc:
        logger.error(f"Failed to initialize CLI: {exc}")
        write_line("Failed to initialize AskMyDocs.")
        return 1

    write_line("AskMyDocs CLI")
    write_line("Type 'quit' to exit.")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except KeyboardInterrupt:
            write_line("\nExiting.")
            return 0
        except EOFError:
            write_line("\nExiting.")
            return 0

        if question.lower() in {"quit", "exit", "q"}:
            write_line("Exiting.")
            return 0
        if not question:
            continue

        try:
            result = rag.ask(question)
        except Exception as exc:
            logger.error(f"Failed to answer CLI question: {exc}")
            write_line("An error occurred while processing the question.")
            continue

        write_line(f"Answer: {result['answer']}")
        write_line(f"Confidence: {result['confidence']}")
        write_line(f"Verified: {result['verified']}")
        if result["sources"]:
            write_line("Sources:")
            for source in result["sources"]:
                write_line(
                    f"- {source['file']} (page {source['page']}, rerank={source['rerank_score']})"
                )


if __name__ == "__main__":
    raise SystemExit(main())
