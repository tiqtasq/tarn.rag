"""Application configuration (pydantic-settings) — env-driven.

``Settings`` is the single source of runtime config; the composition roots (the API's
lifespan in ``app/main.py`` and ``run_worker.py``) read it to build the wiring. The
pipeline/sink *factories* live in the top-level ``config.py`` (they consume a ``Settings``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables (and an optional ``.env``)."""

    # FastAPI
    APP_NAME: str = "RAG Ingestion"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Databases (two separate stores — never conflate them)
    QUEUE_DB_URL: str  # pgQueuer job queue
    DOCUMENT_DB_URL: str  # document / chunk / embedding storage

    # Ingestion
    EMBEDDING_MODEL: str = "sentence-transformers/all-minilm-l6-v2"
    EMBEDDING_DIMENSION: int = 384  # MUST match the model; sets the pgvector column width
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_BATCH_SIZE: int = 32

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
