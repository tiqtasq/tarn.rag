"""Application configuration (pydantic-settings) — env-driven, grouped by component.

``Settings`` is the single source of runtime config; the engines (``IngestionEngine`` /
``RetrievalEngine``) read it in their ``create()`` factories to build the wiring. Config is
**grouped into nested sub-models** (``settings.embedding``, ``settings.index``,
``settings.database`` …) so each component depends only on its slice — env vars use the
``GROUP__FIELD`` convention (e.g. ``EMBEDDING__MODEL``). The few cross-cutting knobs
(``MODE``, ``EMBEDDING_DIMENSION``, ``UPLOAD_DIR``) stay top-level.

The pipeline/sink *factories* live in ``tarnrag/ingestion/factories.py`` (they consume ``Settings``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# How document ids (== source_id) are assigned. The policy is strict — a mismatch fails
# ingestion — so an instance never mixes id schemes; either way the id is **stable** (it never
# changes when a document's content is replaced):
#   'caller' — the caller supplies every source_id (fails if any is missing).
#   'uuid'   — the engine assigns a random uuid4 (fails if the caller supplies any).
# Content dedup is independent of identity: every document also stores a ``content_hash``
# (sha256 of its submitted bytes/text), queryable to detect duplicate or unchanged content.
IdPolicy = Literal["caller", "uuid"]


class AppSettings(BaseModel):
    """
    Process metadata (largely vestigial since the FastAPI layer moved out).
    """

    name: str = "RAG Ingestion"
    version: str = "0.1.0"
    debug: bool = False


class EmbeddingSettings(BaseModel):
    """
    The shared ONNX embedding pipeline (ingestion passages + retrieval queries). Its
    identity (model/revision/prefixes/…) feeds the index fingerprint. ``EMBEDDING_DIMENSION``
    is top-level — it's cross-cutting (index + repo must match it).
    """

    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # model id (recorded in index_meta)
    revision: str = ""
    model_dir: str = "./models/all-MiniLM-L6-v2"  # local model.onnx + tokenizer.json (offline)
    max_seq_length: int = 512
    query_prefix: str = ""  # non-empty for asymmetric models (BGE/E5)
    passage_prefix: str = ""
    batch_size: int = 32  # embed-stage batching


class ChunkingSettings(BaseModel):
    """
    Text chunking for the ingestion pipeline.
    """

    size: int = 512
    overlap: int = 50


class IndexSettings(BaseModel):
    """
    The §8 retrieval index (sqlite-vec/FTS5). Domain fields default until modeled.
    """

    db_path: str = "./index.db"
    default_license_class: str = "public_domain"


class DatabaseSettings(BaseModel):
    """
    The two stores — never conflate them. ``document_url`` defaults to local SQLite so
    embedded mode is zero-config; ``queue_url`` (pgQueuer) is only used in distributed mode.
    """

    document_url: str = "sqlite:///./rag_docs.db"  # document / chunk / embedding storage
    queue_url: str = ""  # pgQueuer job queue (required for MODE='distributed')


class WorkerSettings(BaseModel):
    """
    Distributed-mode worker tuning.
    """

    queue_timeout_seconds: int = 30
    concurrency: int = 4


class ObservabilitySettings(BaseModel):
    """
    Observability toggle (Phase 5). Real adapters plug in behind the ABC.
    """

    enabled: bool = False
    type: str | None = None  # "prometheus" | "structured_logging"


class Settings(BaseSettings):
    """
    Application configuration from environment variables (and an optional ``.env``).

    Grouped fields use the ``GROUP__FIELD`` env convention, e.g. ``EMBEDDING__MODEL``,
    ``DATABASE__DOCUMENT_URL``. The top-level fields keep their flat names.
    """

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    # Execution mode for IngestionEngine. 'embedded' runs the whole pipeline in-process
    # (InMemory queue — no Postgres/pgQueuer needed); 'distributed' enqueues to pgQueuer.
    MODE: Literal["embedded", "distributed"] = "embedded"

    # Cross-cutting: the embedder's output dim — the index and repo MUST match it.
    EMBEDDING_DIMENSION: int = 384

    # Where streamed bytes are staged for the worker (a shared volume in distributed mode).
    UPLOAD_DIR: str = "./uploads"

    # How document ids (== source_id) are assigned/validated — see ``IdPolicy``. Default
    # 'uuid' keeps zero-config ingestion working without the caller managing ids.
    ID_POLICY: IdPolicy = "uuid"

    app: AppSettings = AppSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    index: IndexSettings = IndexSettings()
    database: DatabaseSettings = DatabaseSettings()
    worker: WorkerSettings = WorkerSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @model_validator(mode="after")
    def _check_database_for_mode(self) -> Settings:
        """Fail fast on a MODE / document-store mismatch — ``MODE`` pins the backend:
        distributed needs Postgres (+ pgQueuer); embedded needs SQLite (single-process). This
        catches both a distributed deploy that forgot to point at Postgres (would silently use
        the SQLite default) and an embedded run aimed at Postgres (probably a leftover/typo)."""
        is_postgres = "postgres" in self.database.document_url
        if self.MODE == "distributed":
            if not self.database.queue_url:
                raise ValueError(
                    "MODE='distributed' requires DATABASE__QUEUE_URL (the pgQueuer job queue)"
                )
            if not is_postgres:
                raise ValueError(
                    "MODE='distributed' requires a Postgres DATABASE__DOCUMENT_URL; got "
                    f"{self.database.document_url!r} (SQLite is single-process — embedded only)"
                )
        elif is_postgres:  # embedded
            raise ValueError(
                "MODE='embedded' must not use a Postgres DATABASE__DOCUMENT_URL; got "
                f"{self.database.document_url!r} (use SQLite, e.g. 'sqlite:///./rag_docs.db', "
                "or switch to MODE='distributed')"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings — built once per process from the environment."""
    return Settings()
