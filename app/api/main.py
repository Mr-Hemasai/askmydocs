# askmydocs/app/api/main.py — FastAPI application for ingestion, querying, and document management
from __future__ import annotations

import time
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.chain import DefensiveRAG
from app.vectorstore.store import delete_document, get_indexed_documents, ingest_documents

logger = get_logger()
app = FastAPI(title="AskMyDocs", version="2.0.0")
sessions: dict[str, DefensiveRAG] = {}


class AskRequest(BaseModel):
    """Request body for question answering."""

    question: str = Field(..., min_length=1)
    session_id: str | None = None


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log request and response timing."""

    started_at = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} in {duration_ms:.2f}ms")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize persistent indexes on startup."""

    try:
        ingest_documents()
    except Exception as exc:
        logger.error(f"Server error during startup ingestion: {exc}")


def get_session(session_id: str | None) -> tuple[str, DefensiveRAG]:
    """Return an existing session or create a new one."""

    resolved_session_id = session_id or str(uuid4())
    if resolved_session_id not in sessions:
        sessions[resolved_session_id] = DefensiveRAG()
    return resolved_session_id, sessions[resolved_session_id]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured 500 response for unhandled errors."""

    logger.error(f"Server error on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, object]:
    """Save uploaded PDFs and trigger incremental ingestion."""

    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        destination = settings.documents_dir / Path(upload.filename).name
        try:
            content = await upload.read()
            destination.write_bytes(content)
        except OSError as exc:
            logger.error(f"File read error while saving {upload.filename}: {exc}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
        saved_files.append(destination.name)

    result = ingest_documents()
    sessions.clear()
    return {
        "status": "ok",
        "files_added": saved_files,
        "chunks_indexed": result.chunks_indexed,
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, object]:
    """Answer a question using a session-scoped defensive RAG engine."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    session_id, rag = get_session(request.session_id)
    try:
        answer = rag.ask(request.question)
    except Exception as exc:
        logger.error(f"Server error while answering question: {exc}")
        raise HTTPException(status_code=500, detail="Failed to answer question.") from exc

    answer["session_id"] = session_id
    return answer


@app.get("/documents")
async def list_documents() -> list[dict[str, object]]:
    """List indexed document metadata."""

    return get_indexed_documents()


@app.delete("/documents/{filename}")
async def remove_document(filename: str) -> dict[str, str]:
    """Delete a document from disk and indexes."""

    try:
        delete_document(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except Exception as exc:
        logger.error(f"Server error while deleting {filename}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to delete document.") from exc

    sessions.clear()
    return {"status": "ok", "removed": filename}


@app.get("/health")
async def health() -> dict[str, object]:
    """Return service health information."""

    return {
        "status": "ok",
        "model": settings.OLLAMA_MODEL,
        "documents_indexed": len(get_indexed_documents()),
        "vectordb": "chroma",
    }
