from datetime import UTC, datetime

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.engine.jobs import IngestionJob
from tarnrag.ingestion.engine.queue import InMemoryJobQueue


def _job(job_id):
    return IngestionJob(
        job_id=job_id,
        document_id="s1",
        item=PipelineItem(content="x", metadata={"source_id": "s1"}),
        stage_name="LoadAndParse",
        created_at=datetime.now(UTC),
    )


async def test_processes_and_drains_newly_enqueued():
    q = InMemoryJobQueue()
    seen: list[str] = []

    async def handler(batch):
        for j in batch.jobs:
            seen.append(j.job_id)
            if j.job_id == "j1":  # emulate fan-out
                await q.enqueue(_job("child"))

    q.set_handler(handler)
    await q.enqueue(_job("j1"))
    await q.run()
    assert set(seen) == {"j1", "child"}  # drains jobs enqueued during handling


async def test_requeue_then_dead_letter():
    q = InMemoryJobQueue(max_attempts=2)
    attempts: list[str] = []

    async def handler(batch):
        attempts.append(batch.jobs[0].job_id)
        raise RuntimeError("boom")

    q.set_handler(handler)
    await q.enqueue(_job("j1"))
    await q.run()
    assert len(attempts) == 2  # retried up to max_attempts
    assert [j.job_id for j in q.dead_letters] == ["j1"]


async def test_raise_through_mode():
    q = InMemoryJobQueue(requeue_on_error=False)

    async def handler(batch):
        raise ValueError("x")

    q.set_handler(handler)
    await q.enqueue(_job("j1"))
    with pytest.raises(ValueError):
        await q.run()
