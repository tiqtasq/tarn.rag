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

from app.api.v1.dependencies import build_service, make_queue, make_repository
from app.api.v1.endpoints import ingestion
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repository = await make_repository(settings)
    enqueuer = await make_queue(settings)
    app.state.ingestion_service = build_service(settings, enqueuer, repository)
    logger.info("Ingestion API ready (%s v%s)", settings.APP_NAME, settings.APP_VERSION)
    try:
        yield
    finally:
        await repository.engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API app. Settings (which require env) are read lazily in the lifespan, so
    constructing the app itself needs no environment — handy for tests, which override the
    service dependency and never start the real lifespan."""
    name, version = ("RAG Ingestion", "0.1.0")
    if settings is not None:
        name, version = settings.APP_NAME, settings.APP_VERSION
    app = FastAPI(title=name, version=version, lifespan=lifespan)
    app.include_router(ingestion.router)
    return app


app = create_app()
