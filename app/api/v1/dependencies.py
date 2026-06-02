"""Dependency injection + composition builders.

The wiring is built **once per process** (the API's lifespan in ``app/main.py`` and
``run_worker.py``) via the ``make_*`` / ``build_*`` helpers — NOT per request. (The repo
owns a connection pool; rebuilding it per request would reconnect every call and lose a
SQLite in-memory DB.) The request-scoped dependency just reads the singleton off
``app.state``; tests override it with an InMemory + SQLite wiring.
"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings, get_settings
from app.domains.base.repository import DocumentRepository
from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.queue import JobEnqueuer, PgQueuerJobQueue
from app.domains.ingestion.service import IngestionService
from app.factories import create_ingestion_pipeline, create_sink_registry

__all__ = [
    "get_settings",
    "make_repository",
    "make_queue",
    "build_orchestrator",
    "build_service",
    "get_ingestion_service",
]


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
    settings: Settings, enqueuer: JobEnqueuer, repository: DocumentRepository
) -> PipelineOrchestrator:
    """Assemble the orchestrator (DAG + sinks) — the worker's BatchCoordinator and the
    service's queueing engine."""
    pipeline = create_ingestion_pipeline(settings)
    return PipelineOrchestrator(
        PipelineDAG(pipeline.stages), enqueuer, repository, create_sink_registry()
    )


def build_service(
    settings: Settings, enqueuer: JobEnqueuer, repository: DocumentRepository
) -> IngestionService:
    """Assemble the full facade for the API process."""
    pipeline = create_ingestion_pipeline(settings)
    orchestrator = PipelineOrchestrator(
        PipelineDAG(pipeline.stages), enqueuer, repository, create_sink_registry()
    )
    return IngestionService(pipeline, orchestrator, repository)


def get_ingestion_service(request: Request) -> IngestionService:
    """Request-scoped dependency: the process-wide service built in the app lifespan."""
    return request.app.state.ingestion_service
