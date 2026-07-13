from datetime import UTC, datetime

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.engine.jobs import IngestionJob
from tarnrag.ingestion.engine.queue import InMemoryJobQueue


def _job(job_id, stage="LoadAndParse"):
    return IngestionJob(
        job_id=job_id,
        document_id="s1",
        item=PipelineItem(content="x", metadata={"source_id": "s1"}),
        stage_name=stage,
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


async def test_batches_same_stage_jobs_in_one_dispatch():
    q = InMemoryJobQueue()
    sizes: list[int] = []

    async def handler(batch):
        sizes.append(len(batch.jobs))

    q.set_handler(handler)
    for i in range(5):
        await q.enqueue(_job(f"j{i}"))
    await q.run()
    assert sizes == [5]  # all five same-stage jobs ride one batch (not one-per-dispatch)


async def test_groups_a_wave_by_stage_into_homogeneous_batches():
    q = InMemoryJobQueue()
    seen: list[tuple[str, int]] = []

    async def handler(batch):
        seen.append((batch.stage_name, len(batch.jobs)))

    q.set_handler(handler)
    await q.enqueue(_job("a", stage="LoadAndParse"))
    await q.enqueue(_job("b", stage="Embed"))
    await q.enqueue(_job("c", stage="LoadAndParse"))
    await q.run()
    assert sorted(seen) == [("Embed", 1), ("LoadAndParse", 2)]  # one homogeneous batch per stage


async def test_max_batch_size_caps_the_dispatch():
    q = InMemoryJobQueue(max_batch_size=2)
    sizes: list[int] = []

    async def handler(batch):
        sizes.append(len(batch.jobs))

    q.set_handler(handler)
    for i in range(5):
        await q.enqueue(_job(f"j{i}"))
    await q.run()
    assert sorted(sizes, reverse=True) == [2, 2, 1]  # capped at max_batch_size


async def test_batched_failure_isolates_the_culprit():
    q = InMemoryJobQueue(max_attempts=2)
    processed: list[str] = []

    async def handler(batch):
        for j in batch.jobs:
            if j.job_id == "bad":
                raise RuntimeError("boom")  # fails the whole batch -> re-run each solo to isolate
            processed.append(j.job_id)

    q.set_handler(handler)
    for jid in ("g1", "bad", "g2"):
        await q.enqueue(_job(jid))
    await q.run()
    assert [j.job_id for j in q.dead_letters] == ["bad"]  # only the culprit dead-letters
    assert {"g1", "g2"} <= set(processed)  # the innocents still complete (per-job semantics preserved)


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
