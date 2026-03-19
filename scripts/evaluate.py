# askmydocs/scripts/evaluate.py — Offline RAGAS evaluation harness for AskMyDocs
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.chain import DefensiveRAG
from app.vectorstore.embeddings import get_embedding_model

logger = get_logger()


def write_line(message: str) -> None:
    """Write a line to stdout without using print."""

    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def load_eval_dataset(dataset_path: Path | None = None) -> list[dict[str, str]]:
    """Load question-answer pairs for evaluation."""

    target = dataset_path or settings.eval_dataset_path
    return json.loads(target.read_text(encoding="utf-8"))


def build_eval_rows(rag: DefensiveRAG, qa_pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run the RAG pipeline over the evaluation dataset."""

    rows: list[dict[str, Any]] = []
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        rewritten_query = rag.query_rewriter.rewrite(question)
        retrieval = rag.retriever.retrieve(rewritten_query)
        reranked_docs = rag.reranker.rerank(question, retrieval.documents)
        answer = rag.ask(question)
        rows.append(
            {
                "question": question,
                "answer": answer["answer"],
                "contexts": [document.page_content for document in reranked_docs],
                "ground_truth": qa_pair["ground_truth"],
            }
        )
    return rows


def write_report(payload: dict[str, Any]) -> None:
    """Persist the evaluation report to disk."""

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.ragas_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Run the offline RAGAS evaluation workflow."""

    qa_pairs = load_eval_dataset()
    try:
        rag = DefensiveRAG()
        rag.llm.invoke("Reply with OK.")
    except Exception as exc:
        report = {
            "status": "skipped",
            "reason": f"Ollama model unavailable: {exc}",
            "dataset_size": len(qa_pairs),
        }
        write_report(report)
        logger.error(f"Evaluation skipped: {exc}")
        write_line(json.dumps(report, indent=2))
        return 0

    rows = build_eval_rows(rag, qa_pairs)
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=LangchainLLMWrapper(rag.llm),
        embeddings=LangchainEmbeddingsWrapper(get_embedding_model()),
    )
    report = {
        "status": "ok",
        "dataset_size": len(rows),
        "metrics": result.to_pandas().to_dict(orient="records")[0],
    }
    write_report(report)
    write_line(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
