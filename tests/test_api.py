# askmydocs/tests/test_api.py — Tests for FastAPI endpoints with mocked RAG sessions
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import main as api_main


class FakeRAG:
    """Simple RAG stub for API tests."""

    def ask(self, question: str) -> dict[str, object]:
        """Return a deterministic API response."""

        return {
            "answer": f"Echo: {question}",
            "confidence": "High",
            "sources": [{"file": "doc.pdf", "page": 1, "chunk_id": "c1", "rerank_score": 1.0}],
            "turn_number": 1,
            "verified": True,
        }


@pytest.mark.asyncio
async def test_api_endpoints(monkeypatch, isolated_paths: dict[str, Path]) -> None:
    """Exercise the main FastAPI endpoints."""

    api_main.sessions.clear()
    monkeypatch.setattr(api_main, "ingest_documents", lambda: type("Result", (), {"chunks_indexed": 3})())
    monkeypatch.setattr(api_main, "get_indexed_documents", lambda: [{"filename": "doc.pdf", "hash": "abc", "chunk_count": 3, "file_size": 100, "date_added": "2024-01-01T00:00:00+00:00"}])
    monkeypatch.setattr(api_main, "delete_document", lambda filename: None)
    monkeypatch.setattr(api_main, "DefensiveRAG", FakeRAG)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        ingest_response = await client.post(
            "/ingest",
            files={"files": ("doc.pdf", b"%PDF-1.4 test", "application/pdf")},
        )
        ask_response = await client.post("/ask", json={"question": "What is this?"})
        documents_response = await client.get("/documents")
        delete_response = await client.delete("/documents/doc.pdf")
        health_response = await client.get("/health")

    assert ingest_response.status_code == 200
    assert ask_response.status_code == 200
    assert "session_id" in ask_response.json()
    assert documents_response.status_code == 200
    assert delete_response.status_code == 200
    assert health_response.json()["vectordb"] == "chroma"
