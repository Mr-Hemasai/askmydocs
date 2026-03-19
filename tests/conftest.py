# askmydocs/tests/conftest.py — Shared pytest fixtures for AskMyDocs
from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.core.config import settings


@pytest.fixture()
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect configurable storage paths into a temporary test directory."""

    documents_dir = tmp_path / "documents"
    vector_db_dir = tmp_path / "vector_db"
    documents_dir.mkdir(parents=True, exist_ok=True)
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "DOCUMENTS_PATH", str(documents_dir))
    monkeypatch.setattr(settings, "VECTOR_DB_PATH", str(vector_db_dir))
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_VERIFICATION", False)
    return {"documents_dir": documents_dir, "vector_db_dir": vector_db_dir}


@pytest.fixture()
def synthetic_pdf(isolated_paths: dict[str, Path]) -> Path:
    """Create a synthetic PDF document with test content."""

    pdf_path = isolated_paths["documents_dir"] / "synthetic.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 750, "Artificial intelligence helps solve real-world problems.")
    pdf.drawString(72, 730, "Hybrid retrieval combines dense search and keyword search.")
    pdf.save()
    return pdf_path
