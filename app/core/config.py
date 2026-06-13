"""Application configuration (pydantic-settings) — env-driven.

``Settings`` is the single source of runtime config; the engines (``IngestionEngine`` /
``RetrievalEngine``) read it in their ``create()`` factories to build the wiring. The
pipeline/sink *factories* live in ``app/factories.py`` (they consume a ``Settings``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables (and an optional ``.env``)."""

    # FastAPI
    APP_NAME: str = "RAG Ingestion"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Execution mode for IngestionEngine. 'embedded' runs the whole pipeline in-process
    # (InMemory queue — no Postgres/pgQueuer needed), the easy single-process path;
    # 'distributed' enqueues to pgQueuer for separate worker processes to consume.
    MODE: Literal["embedded", "distributed"] = "embedded"

    # Databases (two separate stores — never conflate them)
    QUEUE_DB_URL: str = ""  # pgQueuer job queue (required for MODE='distributed')
    DOCUMENT_DB_URL: str  # document / chunk / embedding storage

    # Ingestion / embedding (ONNX). The model is configurable; ingestion and retrieval MUST
    # share it — enforced by the embedding_config_fingerprint recorded in the index.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # model id (recorded in index_meta)
    EMBEDDING_MODEL_REVISION: str = ""
    MODEL_DIR: str = "./models/all-MiniLM-L6-v2"  # local model.onnx + tokenizer.json (offline)
    EMBEDDING_DIMENSION: int = 384  # MUST match the model
    MAX_SEQ_LENGTH: int = 512
    EMBEDDING_QUERY_PREFIX: str = ""  # non-empty for asymmetric models (BGE/E5)
    EMBEDDING_PASSAGE_PREFIX: str = ""
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_BATCH_SIZE: int = 32

    # §8 retrieval index (sqlite-vec/FTS5). Domain fields default until modeled.
    INDEX_DB_PATH: str = "./index.db"
    DEFAULT_LICENSE_CLASS: str = "public_domain"

    # Uploads: where the API stages uploaded bytes so workers can read them by path.
    # Must be a location both the API and worker processes can access (shared volume).
    UPLOAD_DIR: str = "./uploads"

    # Workers
    WORKER_QUEUE_TIMEOUT_SECONDS: int = 30
    WORKER_CONCURRENCY: int = 4

    # Observability (Phase 5)
    OBSERVABILITY_ENABLED: bool = False
    OBSERVABILITY_TYPE: str | None = None  # "prometheus" | "structured_logging"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings — built once per process from the environment."""
    return Settings()
