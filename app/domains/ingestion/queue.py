"""Job queue ports + adapters.

The queue surface is segregated by role: ``JobEnqueuer`` (producer — the orchestrator
puts jobs on the queue) and ``JobConsumer`` (the runtime — register a handler and run
the dispatch loop, which belongs to the queue technology). One adapter implements both:
``PgQueuerJobQueue`` (the only place that touches pgQueuer, lazily imported) and
``InMemoryJobQueue`` (a test double that runs the whole flow with no Postgres/pgQueuer).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from app.domains.ingestion.models import Batch, IngestionJob

ENTRYPOINT = "ingest"  # single pgQueuer entrypoint; the consumer forms homogeneous Batches

# A handler receives a Batch — a homogeneous unit (all jobs share one stage_name). Today
# every dispatch is a single job; batch dispatch must group claims by stage_name (Batch
# enforces it). Forming the Batch is the consumer's job; the worker just runs it.
JobHandler = Callable[[Batch], Awaitable[None]]


class JobEnqueuer(ABC):
    """Producer side — put jobs on the queue. Used by the orchestrator."""

    @abstractmethod
    async def enqueue(self, job: IngestionJob) -> None: ...


class JobConsumer(ABC):
    """Consumer/runtime side — register a handler and run the dispatch loop. The loop
    itself belongs to the queue technology (e.g. pgQueuer's QueueManager); the worker
    is just a handler, and the composition root calls ``set_handler`` then ``run``.
    Mechanics (claiming, retries, NOTIFY, dead-lettering) belong to the implementation."""

    @abstractmethod
    def set_handler(self, handler: JobHandler) -> None: ...

    @abstractmethod
    async def run(self) -> None: ...


class InMemoryJobQueue(JobEnqueuer, JobConsumer):
    """In-process job queue for tests — no Postgres, no pgQueuer. Emulates pgQueuer's
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
                await self._handler(Batch([job]))  # one job per dispatch today
            except Exception:
                if not self.requeue_on_error:
                    raise
                if attempts + 1 >= self.max_attempts:
                    self.dead_letters.append(job)
                else:
                    await self._jobs.put((job, attempts + 1))


class PgQueuerJobQueue(JobEnqueuer, JobConsumer):
    """Queue backed by pgQueuer — the only adapter that imports pgQueuer (lazily).
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
            await handler(Batch([IngestionJob.model_validate_json(pg_job.payload)]))

    async def run(self) -> None:
        await self._qm.run()
