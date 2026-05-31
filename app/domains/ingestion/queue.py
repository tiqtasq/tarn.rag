"""Job queue port + adapters.

The domain depends only on the tiny ``JobQueue`` port. ``PgQueuerJobQueue`` is the
only place that touches pgQueuer (lazily imported); ``InMemoryJobQueue`` is a test
double that runs the whole flow with no Postgres and no pgQueuer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from app.domains.ingestion.models import IngestionJob

ENTRYPOINT = "ingest"  # single pgQueuer entrypoint; the handler dispatches by stage_name

# A handler receives a BATCH of jobs (one job, or several under batch dispatch).
JobHandler = Callable[[list[IngestionJob]], Awaitable[None]]


@runtime_checkable
class JobQueue(Protocol):
    """The minimal queue surface the domain uses. Mechanics (claiming, retries,
    NOTIFY, dead-lettering) belong to the implementation — never reimplement them."""

    async def enqueue(self, job: IngestionJob) -> None: ...
    def set_handler(self, handler: JobHandler) -> None: ...
    async def run(self) -> None: ...


class InMemoryJobQueue:
    """In-process JobQueue for tests — no Postgres, no pgQueuer. Emulates pgQueuer's
    at-least-once + requeue-on-raise semantics so tests reflect reality.

    ``requeue_on_error=True`` (default) re-queues a failing job up to ``max_attempts``
    (recovery tests); ``False`` re-raises immediately (sharp unit failures). pgQueuer
    does NOT guarantee ordering — tests must not rely on it.
    """

    def __init__(self, requeue_on_error: bool = True, max_attempts: int = 3):
        self._jobs: asyncio.Queue[tuple[IngestionJob, int]] = asyncio.Queue()
        self._handler: JobHandler | None = None
        self.requeue_on_error = requeue_on_error
        self.max_attempts = max_attempts
        self.dead_letters: list[IngestionJob] = []

    async def enqueue(self, job: IngestionJob) -> None:
        await self._jobs.put((job, 0))

    def set_handler(self, handler: JobHandler) -> None:
        self._handler = handler

    async def run(self) -> None:
        """Drain all jobs (including any enqueued while handling) and stop — so tests
        terminate. Use as the test entrypoint instead of a forever loop."""
        assert self._handler is not None, "set_handler() before run()"
        while not self._jobs.empty():
            job, attempts = await self._jobs.get()
            try:
                await self._handler([job])
            except Exception:
                if not self.requeue_on_error:
                    raise
                if attempts + 1 >= self.max_attempts:
                    self.dead_letters.append(job)
                else:
                    await self._jobs.put((job, attempts + 1))


class PgQueuerJobQueue:
    """JobQueue backed by pgQueuer — the only adapter that imports pgQueuer (lazily).
    pgQueuer owns claiming/retries/NOTIFY/dead-lettering; a handler that raises
    propagates so pgQueuer requeues the job (= recovery, D5)."""

    def __init__(self, driver):
        from pgqueuer import QueueManager
        from pgqueuer.queries import Queries

        self._qm = QueueManager(driver)
        self._queries = Queries(driver)

    @classmethod
    async def connect(cls, connection_url: str) -> "PgQueuerJobQueue":
        import asyncpg
        from pgqueuer.db import AsyncpgPoolDriver

        pool = await asyncpg.create_pool(connection_url)
        return cls(AsyncpgPoolDriver(pool))

    async def enqueue(self, job: IngestionJob) -> None:
        # payload is the full IngestionJob (incl. the inline item, D1)
        await self._queries.enqueue([ENTRYPOINT], [job.model_dump_json().encode()], [0])

    def set_handler(self, handler: JobHandler) -> None:
        from pgqueuer.models import Job

        @self._qm.entrypoint(ENTRYPOINT)
        async def _run(pg_job: "Job") -> None:
            await handler([IngestionJob.model_validate_json(pg_job.payload)])

    async def run(self) -> None:
        await self._qm.run()
