"""Application configuration (pydantic-settings) — env-driven, grouped by component.

``Settings`` is the single source of runtime config; the engines (``IngestionEngine`` /
``RetrievalEngine``) read it in their ``create()`` factories to build the wiring. Config is
**grouped into nested sub-models** (``settings.embedding``, ``settings.database`` …) so each
component depends only on its slice — env vars use the ``GROUP__FIELD`` convention (e.g.
``EMBEDDING__MODEL``). The few cross-cutting knobs (``MODE``, ``EMBEDDING_DIMENSION``,
``UPLOAD_DIR``) stay top-level.

Pipeline composition lives in ``IngestionEngine.build_pipeline`` (it consumes ``Settings``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

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

# Key under ``Settings.components`` holding the ingestion pipeline spec (a Pipeline-component spec).
INGESTION_PIPELINE = "ingestion_pipeline"
RETRIEVAL_PIPELINE = "retrieval_pipeline"


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

    # Backend: 'onnx' = local ONNX (default, offline); 'openai' / 'voyage' / 'gemini' = HTTP embedding
    # APIs (the ``embeddings-api`` extra). The provider + model + dimension are part of the index
    # fingerprint, so a locally-built index won't ``open()`` against an API embedder (and vice versa).
    provider: Literal["onnx", "openai", "voyage", "gemini"] = "onnx"

    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # HF id (onnx) or API model name
    revision: str = ""  # onnx only (recorded in index_meta)
    model_dir: str = "./models/all-MiniLM-L6-v2"  # onnx only: local model.onnx + tokenizer.json
    max_seq_length: int = 512  # onnx only
    # onnx pooling over the token outputs: 'mean' (encoder models: MiniLM/BGE/E5) | 'last' (decoder
    # models, e.g. Qwen3-Embedding — last non-pad token) | 'cls'. 'normalize': 'l2' | 'none'.
    pooling: str = "mean"
    normalize: str = "l2"
    query_prefix: str = ""  # prepended to queries (asymmetric models: BGE/E5/Qwen)
    passage_prefix: str = ""  # prepended to passages
    batch_size: int = 32  # embed-stage / API-request batching
    # Header-path injection: prepend each chunk's section header path to its text before embedding
    # (the embed stage applies it; queries are never injected). Part of the embedding identity, so an
    # injected index has a distinct fingerprint and won't ``open()`` with a non-injecting embedder —
    # the two are compared by building separate indexes.
    inject_header_path: bool = False

    # API providers only (provider != 'onnx'):
    api_key: str = ""  # falls back to the provider's standard env var (OPENAI_API_KEY / VOYAGE_API_KEY / GEMINI_API_KEY)
    api_base_url: str = ""  # falls back to the provider's default endpoint
    api_timeout: float = 30.0


class RerankSettings(BaseModel):
    """
    The optional cross-encoder reranker (query × passage relevance). Retrieval-only and only loaded
    when a ``cross_encoder`` reranker is in the ``RETRIEVAL_PIPELINE``; the model loads lazily on first
    use (like the embedder), so a Settings without the model dir present on disk is still valid.
    """

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # model id (for eval / result transparency)
    revision: str = ""
    model_dir: str = "./models/ms-marco-MiniLM-L-6-v2"  # local model.onnx + tokenizer.json (offline)
    max_seq_length: int = 512


class DatabaseSettings(BaseModel):
    """
    The two stores — never conflate them. ``document_url`` is the repository (documents, chunks,
    the §8 retrieval index, and job_status); it defaults to local SQLite so embedded mode is
    zero-config. ``queue_url`` (pgQueuer) is only used in distributed mode.
    """

    document_url: str = "sqlite:///./rag_docs.db"  # repository: docs/chunks/§8 index + job_status
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
    rerank: RerankSettings = RerankSettings()
    database: DatabaseSettings = DatabaseSettings()
    worker: WorkerSettings = WorkerSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    # Named component specs that build pluggable parts of the system — e.g. the ingestion pipeline
    # under ``INGESTION_PIPELINE``. Each value is a raw spec, validated when built via
    # ``ComponentFactory`` (not here). ``_fill_default_components`` ensures the ingestion pipeline is
    # always present (default below), so a Settings is self-complete and consumers read it directly.
    components: dict[str, Any] = {}

    @model_validator(mode="after")
    def _fill_default_components(self) -> Settings:
        """Make a Settings self-complete: ensure the ingestion pipeline spec is present (a
        user-supplied one wins), so consumers read ``components[INGESTION_PIPELINE]`` directly
        instead of recomputing a default elsewhere. Component config (chunker, extractor routes, …)
        lives on the stage specs here — override the pipeline spec to customise it."""
        self.components.setdefault(
            INGESTION_PIPELINE,
            {
                "class_name": "pipeline",
                "stages": [
                    {"class_name": "LoadAndParse"},
                    {"class_name": "Enrich"},  # doc-phase enrichers (default: none) annotate the document
                    {"class_name": "CleanAndNormalize"},
                    {"class_name": "Chunk"},  # chunker defaults to structure_aware on ChunkStage.Config
                    {"class_name": "Embed"},
                ],
            },
        )
        # The retrieval composition (the analog of INGESTION_PIPELINE): dense-only by default; set
        # ``retrievers`` + a ``fuser: {"class_name": "rrf"}`` for hybrid. Comparing methods = vary this.
        self.components.setdefault(
            RETRIEVAL_PIPELINE,
            {
                "class_name": "retrieval_pipeline",
                "retrievers": [{"class_name": "dense"}],
                "fuser": {"class_name": "identity"},
            },
        )
        return self

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
