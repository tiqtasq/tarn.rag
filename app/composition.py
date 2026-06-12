"""Composition builders — the process composition root.

The wiring is built **once per process** in ``run_worker.py`` via the ``make_*`` / ``build_*``
helpers — NOT per request. (The repo owns a connection pool; rebuilding it per request would
reconnect every call and lose a SQLite in-memory DB.)
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.observability import NoOpObservability, Observability
from app.domains.base.embedder import Embedder, OnnxEmbedder
from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.repository import DocumentRepository
from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.queue import JobEnqueuer, PgQueuerJobQueue
from app.domains.ingestion.service import IngestionService
from app.domains.retrieval.engine import RetrievalEngine
from app.factories import create_ingestion_pipeline, create_sink_registry

__all__ = [
    "get_settings",
    "make_repository",
    "make_queue",
    "make_embedder",
    "make_index_store",
    "get_observability",
    "build_orchestrator",
    "build_service",
]


def get_observability(settings: Settings) -> Observability | None:
    """The process observability, or ``None`` when disabled (call sites guard ``self.obs``).
    When enabled we return ``NoOpObservability`` for now — real adapters (Prometheus,
    structured logging) are future work and plug in here behind the same ABC."""
    return NoOpObservability() if settings.OBSERVABILITY_ENABLED else None


def make_embedder(settings: Settings) -> Embedder:
    """The shared ONNX embedder from config (loads model.onnx + tokenizer.json lazily)."""
    return OnnxEmbedder(
        settings.MODEL_DIR,
        model_id=settings.EMBEDDING_MODEL,
        revision=settings.EMBEDDING_MODEL_REVISION,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        max_length=settings.MAX_SEQ_LENGTH,
        query_prefix=settings.EMBEDDING_QUERY_PREFIX,
        passage_prefix=settings.EMBEDDING_PASSAGE_PREFIX,
    )


def make_index_store(settings: Settings, embedder: Embedder | None = None) -> SqliteIndexStore:
    """Open the §8 retrieval index (connects, ensures schema). The worker passes an
    ``embedder`` so it records ``index_meta`` (it's the producer); the API opens read-only
    (no embedder, no model needed)."""
    store = SqliteIndexStore(
        settings.INDEX_DB_PATH,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        default_license_class=settings.DEFAULT_LICENSE_CLASS,
    ).connect()
    if embedder is not None:
        store.write_index_meta(embedder)
    return store


async def make_repository(settings: Settings) -> DocumentRepository:
    """Build and connect the document repository (Postgres on a ``postgres`` URL, else
    SQLite). Heavy backends are imported lazily so the SQLite path stays dependency-light."""
    if "postgres" in settings.DOCUMENT_DB_URL:
        from app.domains.base.postgres_repository import PostgresRepository

        repo: DocumentRepository = PostgresRepository(
            settings.DOCUMENT_DB_URL, embedding_dimension=settings.EMBEDDING_DIMENSION
        )
    else:
        from app.domains.base.sqlite_repository import SqliteRepository

        repo = SqliteRepository(
            settings.DOCUMENT_DB_URL, embedding_dimension=settings.EMBEDDING_DIMENSION
        )
    await repo.connect()
    return repo


async def make_queue(settings: Settings) -> PgQueuerJobQueue:
    """The pgQueuer-backed queue adapter (implements both ports). pgQueuer is imported
    lazily inside ``connect``."""
    return await PgQueuerJobQueue.connect(settings.QUEUE_DB_URL)


def build_orchestrator(
    settings: Settings,
    enqueuer: JobEnqueuer,
    repository: DocumentRepository,
    observability: Observability | None = None,
    index_store: SqliteIndexStore | None = None,
) -> PipelineOrchestrator:
    """Assemble the orchestrator (DAG + sinks) — the worker's BatchCoordinator and the
    service's queueing engine. With ``index_store``, sinks persist into the §8 index (data);
    ``repository`` keeps ``job_status``."""
    pipeline = create_ingestion_pipeline(settings)
    return PipelineOrchestrator(
        PipelineDAG(pipeline.stages), enqueuer, repository, create_sink_registry(),
        observability=observability, chunk_store=index_store,
    )


def build_service(
    settings: Settings,
    enqueuer: JobEnqueuer,
    repository: DocumentRepository,
    observability: Observability | None = None,
    index_store: SqliteIndexStore | None = None,
) -> IngestionService:
    """Assemble the full facade for the API process. With ``index_store``, status reads its
    chunk/embedding counts (facts) from the index while job_status stays on the repository."""
    pipeline = create_ingestion_pipeline(settings)
    orchestrator = PipelineOrchestrator(
        PipelineDAG(pipeline.stages), enqueuer, repository, create_sink_registry(),
        observability=observability, chunk_store=index_store,
    )
    return IngestionService(
        pipeline, orchestrator, repository, observability,
        staging_dir=settings.UPLOAD_DIR, facts_source=index_store,
    )


def build_retrieval_engine(
    settings: Settings, index_store: SqliteIndexStore, embedder: Embedder
) -> RetrievalEngine:
    """Open a query-ready engine over the index. Validates the embedding fingerprint +
    schema against ``index_meta`` (raises ``RetrievalError`` on mismatch / unbuilt index)."""
    return RetrievalEngine.open(index_store, embedder, config=settings)
