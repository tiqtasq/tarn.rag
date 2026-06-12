"""Worker process composition root (the consumer side).

Builds the same wiring as the API (shared ``make_*`` / ``build_*`` helpers), then registers
the pure worker as the queue's handler and runs the consumer loop. Compute happens here;
the orchestrator (the worker's ``BatchCoordinator``) records status, persists via the
sinks, and enqueues downstream jobs. Run multiple instances for parallelism — pgQueuer
owns claiming / SKIP LOCKED / retries.

    python run_worker.py
"""

from __future__ import annotations

import asyncio
import logging

from app.composition import (
    build_orchestrator,
    get_observability,
    make_embedder,
    make_index_store,
    make_queue,
    make_repository,
)
from app.core.config import get_settings
from app.domains.ingestion.worker import IngestionWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    repository = await make_repository(settings)
    queue = await make_queue(settings)
    observability = get_observability(settings)
    # The worker is the index producer: it writes index_meta (embedding fingerprint) and the
    # sinks persist documents/chunks/vectors into the §8 index. job_status stays on the repo.
    index_store = make_index_store(settings, embedder=make_embedder(settings))
    orchestrator = build_orchestrator(
        settings, queue, repository, observability, index_store=index_store
    )

    # pure handler; coordinator = orchestrator. Worker observes compute; orchestrator the lifecycle.
    worker = IngestionWorker(orchestrator, observability=observability)
    queue.set_handler(worker.handle_batch)  # the only wiring the worker needs
    logger.info("Ingestion worker %s started", worker.worker_id)
    try:
        await queue.run()  # consumer loop (pgQueuer owns claiming/retries/NOTIFY)
    finally:
        await repository.engine.dispose()
        index_store.close()


if __name__ == "__main__":
    asyncio.run(main())
