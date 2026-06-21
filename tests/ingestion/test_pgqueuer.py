"""PgQueuerJobQueue adapter — gated integration round-trip against a real Postgres + pgQueuer.

The distributed queue adapter (``connect`` → pool/driver, ``enqueue`` payload serialization, the entrypoint
that decodes a claimed job into a homogeneous ``Batch``, and the ``run`` dispatch loop) is otherwise
unexercised — the default suite runs the ``InMemoryJobQueue`` double. Set ``TARNRAG_TEST_POSTGRES_URL`` to a
Postgres to run it (the same DB ``test_postgres.py`` uses; pgQueuer installs its own tables)::

    docker run -d -e POSTGRES_PASSWORD=test -p 5433:5432 pgvector/pgvector:pg16
    TARNRAG_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/postgres \
        pytest tests/ingestion/test_pgqueuer.py
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from datetime import UTC, datetime

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.engine.jobs import Batch, IngestionJob
from tarnrag.ingestion.engine.queue import PgQueuerJobQueue

PG_URL = os.environ.get("TARNRAG_TEST_POSTGRES_URL", "")
pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.skipif(
        not PG_URL,
        reason="set TARNRAG_TEST_POSTGRES_URL to a Postgres (pgQueuer installs its own tables there)",
    ),
]


def _job(job_id: str = "j1", stage: str = "LoadAndParse") -> IngestionJob:
    return IngestionJob(
        job_id=job_id,
        document_id="s1",
        item=PipelineItem(content="x", metadata={"source_id": "s1"}),
        stage_name=stage,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
async def pgq():
    """A live ``PgQueuerJobQueue`` on a freshly-installed pgQueuer schema; uninstalled + pool-closed after.
    Uses ``connect`` (so it's exercised) and reaches the driver's pool only to close it cleanly."""
    pytest.importorskip("pgqueuer")
    pytest.importorskip("asyncpg")
    q = await PgQueuerJobQueue.connect(PG_URL)
    with suppress(Exception):
        await q._queries.uninstall()  # clean slate if a previous run left tables behind
    await q._queries.install()
    try:
        yield q
    finally:
        with suppress(Exception):
            await q._queries.uninstall()
        await q._queries.driver._pool.close()  # release the asyncpg pool connect() opened


async def test_enqueued_job_is_consumed_through_the_run_loop(pgq):
    """End-to-end through the public adapter API: enqueue serializes the job, the run loop claims it, and
    the registered entrypoint decodes the payload back into a single-job homogeneous ``Batch``."""
    received: list[Batch] = []
    done = asyncio.Event()

    async def handler(batch: Batch) -> None:
        received.append(batch)
        done.set()

    await pgq.enqueue(_job(stage="Chunk"))
    pgq.set_handler(handler)
    task = asyncio.create_task(pgq.run())  # the real (continuous) dispatch loop
    try:
        await asyncio.wait_for(done.wait(), timeout=30)
    finally:
        task.cancel()  # stop the otherwise-forever loop once the job is handled
        with suppress(asyncio.CancelledError):
            await task

    assert len(received) == 1
    [batch] = received
    assert batch.stage_name == "Chunk"  # decoded from the JSON payload into a homogeneous Batch
    assert batch.jobs[0].document_id == "s1"
    assert batch.jobs[0].item.content == "x"
