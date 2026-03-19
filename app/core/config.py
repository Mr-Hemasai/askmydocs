# askmydocs/app/core/config.py — Centralized application settings loaded from environment variables
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for AskMyDocs."""

    OLLAMA_MODEL: str = Field(default="mistral")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-base-en-v1.5")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    VECTOR_DB_PATH: str = Field(default="./vector_db")
    DOCUMENTS_PATH: str = Field(default="./data/documents")
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=100)
    TOP_K_RETRIEVAL: int = Field(default=10)
    TOP_K_RERANKED: int = Field(default=4)
    DISTANCE_THRESHOLD: float = Field(default=1.2)
    CONFIDENCE_HIGH: float = Field(default=0.80)
    CONFIDENCE_MEDIUM: float = Field(default=0.70)
    ENABLE_HYDE: bool = Field(default=False)
    ENABLE_VERIFICATION: bool = Field(default=False)
    BM25_WEIGHT: float = Field(default=0.4)
    FAISS_WEIGHT: float = Field(default=0.6)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @property
    def base_dir(self) -> Path:
        """Return the project root directory."""

        return Path(__file__).resolve().parents[2]

    @property
    def vector_db_dir(self) -> Path:
        """Return the persistent vector database directory."""

        return (self.base_dir / self.VECTOR_DB_PATH).resolve()

    @property
    def chroma_path(self) -> Path:
        """Return the Chroma persistence directory."""

        return self.vector_db_dir / "chroma"

    @property
    def documents_dir(self) -> Path:
        """Return the PDF documents directory."""

        return (self.base_dir / self.DOCUMENTS_PATH).resolve()

    @property
    def doc_hashes_path(self) -> Path:
        """Return the document hash manifest file path."""

        return self.vector_db_dir / "doc_hashes.json"

    @property
    def bm25_index_path(self) -> Path:
        """Return the BM25 pickle file path."""

        return self.vector_db_dir / "bm25_index.pkl"

    @property
    def document_manifest_path(self) -> Path:
        """Return the indexed document manifest file path."""

        return self.vector_db_dir / "document_manifest.json"

    @property
    def logs_dir(self) -> Path:
        """Return the logs directory."""

        return (self.base_dir / "logs").resolve()

    @property
    def eval_dataset_path(self) -> Path:
        """Return the evaluation dataset file path."""

        return (self.base_dir / "data" / "eval" / "qa_pairs.json").resolve()

    @property
    def ragas_report_path(self) -> Path:
        """Return the RAGAS report file path."""

        return self.logs_dir / "ragas_report.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()
