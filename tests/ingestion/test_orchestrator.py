import pytest

from tarnrag.core.exceptions import IngestionError
from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.engine.jobs import Batch
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.pipeline import MapperStage
from tarnrag.ingestion.queue import JobEnqueuer
from tarnrag.ingestion.result_sink import (
    FinalizationOutcome,
    PassthroughSink,
    ResultSink,
    create_sink_registry,
)


class RecordingEnqueuer(JobEnqueuer):
    """A JobEnqueuer that just records enqueued jobs."""

    def __init__(self):
        self.enqueued: list = []

    async def enqueue(self, job) -> None:
        self.enqueued.append(job)


class _NoopStage(MapperStage):
    def __init__(self, name: str):
        super().__init__(MapperStage.Config(name=name))

    def map(self, text, metadata):
        return text, {}


def _dag(*names):
    return PipelineDAG([_NoopStage(n) for n in names])


def _item(source_id="s1", **meta):
    return PipelineItem(content="x", metadata={"source_id": source_id, **meta})


async def test_ingest_documents_enqueues_root_jobs(repo):
    q = RecordingEnqueuer()
    orch = PipelineOrchestrator(
        _dag("LoadAndParse", "CleanAndNormalize"), q, repo, create_sink_registry()
    )
    ids = await orch.ingest_documents([_item("s1")])
    assert ids == ["s1"]
    assert len(q.enqueued) == 1
    assert q.enqueued[0].stage_name == "LoadAndParse"
    assert q.enqueued[0].document_id == "s1"
    assert (await repo.document_status("s1"))["status"] == "pending"  # queued, no data yet


async def test_complete_fans_out_one_job_per_item(repo):
    q = RecordingEnqueuer()
    # CleanAndNormalize -> Chunk; Clean's sink is PassthroughSink (persists nothing).
    orch = PipelineOrchestrator(_dag("CleanAndNormalize", "Chunk"), q, repo, create_sink_registry())
    job = orch._make_job(_item("s1"), "CleanAndNormalize")
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([_item("s1") for _ in range(3)])  # 3 produced -> 3 downstream Chunk jobs
    await ctx.complete()
    assert len(q.enqueued) == 3
    assert all(j.stage_name == "Chunk" for j in q.enqueued)


async def test_complete_terminal_enqueues_nothing(repo):
    q = RecordingEnqueuer()
    orch = PipelineOrchestrator(_dag("CleanAndNormalize", "Embed"), q, repo, create_sink_registry())
    job = orch._make_job(_item("s1"), "Embed")  # Embed is terminal
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([])  # terminal: no embeddings to persist here, no downstream regardless
    await ctx.complete()
    assert q.enqueued == []


async def test_complete_persistence_failure_raises_and_records(repo):
    q = RecordingEnqueuer()

    class FailingSink(ResultSink):
        def __init__(self, repository):
            pass

        def submit(self, results):
            pass

        def close(self):
            pass

        async def finalize(self):
            return FinalizationOutcome(persisted=False, detail="disk on fire")

    orch = PipelineOrchestrator(
        _dag("X", "Y"), q, repo, {"X": FailingSink, "Y": PassthroughSink}
    )
    job = orch._make_job(_item("s1"), "X")
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([_item("s1")])
    with pytest.raises(IngestionError):
        await ctx.complete()
    assert q.enqueued == []  # no downstream on failure
    assert (await repo.document_status("s1"))["status"] == "failed"


def test_dag_edges_and_terminal():
    dag = _dag("A", "B", "C")
    assert dag.get_downstream_stages("A") == ["B"]
    assert dag.get_downstream_stages("C") == []  # terminal
