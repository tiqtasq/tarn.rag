"""FastAPI app (the API process / composition root for the producer side).

The lifespan builds the wiring ONCE — repository (connected), queue (enqueuer), and the
``IngestionService`` — and stores the service on ``app.state``. The API process only
*enqueues* root jobs; downstream fan-out and persistence happen in the worker process
(see ``run_worker.py``), which shares the same databases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.dependencies import (
    build_retrieval_engine,
    build_service,
    get_observability,
    make_embedder,
    make_index_store,
    make_queue,
    make_repository,
)
from app.api.v1.endpoints import ingestion, retrieval
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repository = await make_repository(settings)
    enqueuer = await make_queue(settings)
    observability = get_observability(settings)
    # The API opens the §8 index read-only (status facts); the worker is the producer/writer.
    index_store = make_index_store(settings)
    app.state.index_store = index_store
    app.state.ingestion_service = build_service(
        settings, enqueuer, repository, observability, index_store=index_store
    )
    # Open the retrieval engine over the same index. Tolerate an unbuilt/incompatible index
    # (cold start before the worker has built it) — /v1/query then returns 503 until it exists.
    try:
        app.state.retrieval_engine = build_retrieval_engine(
            settings, index_store, make_embedder(settings)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Retrieval engine not opened (index not built/compatible yet): %s", e)
        app.state.retrieval_engine = None
    logger.info("Ingestion API ready (%s v%s)", settings.APP_NAME, settings.APP_VERSION)
    try:
        yield
    finally:
        await repository.engine.dispose()
        index_store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API app. Settings (which require env) are read lazily in the lifespan, so
    constructing the app itself needs no environment — handy for tests, which override the
    service dependency and never start the real lifespan."""
    name, version = ("RAG Ingestion", "0.1.0")
    if settings is not None:
        name, version = settings.APP_NAME, settings.APP_VERSION
    app = FastAPI(title=name, version=version, lifespan=lifespan)
    app.include_router(ingestion.router)
    app.include_router(retrieval.router)
    return app


app = create_app()
