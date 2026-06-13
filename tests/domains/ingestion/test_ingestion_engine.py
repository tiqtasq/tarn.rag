"""IngestionEngine facade on real InMemoryJobQueue + SQLite wiring (no API, no Postgres).

Uses a fake 3-d encoder so no sentence-transformers is needed (matches the e2e test). The
bare constructor is used here (the low-level seam); ``create()`` is exercised only when a real
model + index are available.
"""

import io

import pytest

from app.domains.ingestion import DocumentRef, IngestionEngine
from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.pipeline import Pipeline
from app.domains.ingestion.queue import InMemoryJobQueue, JobEnqueuer
from app.domains.ingestion.result_sink import create_sink_registry
from app.domains.ingestion.stages.chunk import ChunkStage
from app.domains.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from app.domains.ingestion.stages.embed import EmbedStage
from app.domains.ingestion.stages.enrich import EnrichMetadataStage
from app.domains.ingestion.stages.load_parse import LoadAndParseStage
from app.domains.ingestion.worker import IngestionWorker


class _FakeEmbedder:
    def embed_passages(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeEmbedStage(EmbedStage):
    def _get_embedder(self):
        return _FakeEmbedder()


def _stages():
    return [
        LoadAndParseStage(),
        CleanAndNormalizeStage(),
        ChunkStage(chunk_size=30, overlap=5),
        EnrichMetadataStage(),
        FakeEmbedStage(model_batch_size=2),
    ]


def _wire(repo, *, auto_drain=False):
    """Build producer+consumer wiring around one repo; return (engine, queue)."""
    queue = InMemoryJobQueue()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)
    engine = IngestionEngine(Pipeline(stages), orch, repo, queue=queue, auto_drain=auto_drain)
    return engine, queue


async def test_ingest_content_queues_then_completes(repo):
    engine, queue = _wire(repo)

    result = await engine.ingest_content([{"content": "Hello world. " * 10, "source_id": "s1"}])
    assert result.queued == 1
    assert result.documents == [DocumentRef("s1")]  # status defaults to "queued"
    # Queued but not yet processed -> pending (no persisted data).
    assert (await engine.status("s1")).status == "pending"

    await queue.run()  # drain the whole DAG

    status = await engine.status("s1")
    assert status.status == "complete"
    assert status.chunk_count >= 1
    assert status.embedding_count == status.chunk_count
    assert status.jobs is None  # not verbose


async def test_embedded_auto_drain_processes_inline(repo):
    """With auto_drain (embedded mode) an ingest call runs the whole pipeline before returning."""
    engine, _ = _wire(repo, auto_drain=True)
    await engine.ingest_content([{"content": "Hello world. " * 10, "source_id": "s1"}])
    # No manual queue.run() — auto_drain processed it inline.
    status = await engine.status("s1")
    assert status.status == "complete"
    assert status.embedding_count == status.chunk_count >= 1


async def test_unknown_document_status_is_none(repo):
    engine, _ = _wire(repo)
    assert await engine.status("does-not-exist") is None


async def test_verbose_adds_jobs_breakdown(repo):
    engine, queue = _wire(repo)
    await engine.ingest_content([{"content": "alpha beta gamma. " * 8, "source_id": "s1"}])
    await queue.run()

    status = await engine.status("s1", verbose=True)
    assert isinstance(status.jobs, list) and status.jobs
    assert all(j["status"] == "completed" for j in status.jobs)


async def test_missing_source_id_gets_assigned(repo):
    engine, _ = _wire(repo)
    result = await engine.ingest_content([{"content": "x"}])
    assert result.queued == 1
    assert result.documents[0].document_id  # a uuid was assigned


class _RecordingEnqueuer(JobEnqueuer):
    def __init__(self):
        self.jobs: list = []

    async def enqueue(self, job) -> None:
        self.jobs.append(job)


async def test_parser_choice_rides_in_item_metadata(repo):
    """The per-request parser selection flows inline on the root job's item (not stage config)."""
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(Pipeline(stages), orch, repo)

    await engine.ingest_content([{"content": "x", "source_id": "s1"}], parser="pdfplumber")
    assert enq.jobs[0].item.metadata["parser"] == "pdfplumber"

    # Omitted -> no parser key (stage falls back to its default).
    enq.jobs.clear()
    await engine.ingest_content([{"content": "x", "source_id": "s2"}])
    assert "parser" not in enq.jobs[0].item.metadata


async def test_ingest_streams_stages_bytes_and_queues_by_path(repo, tmp_path):
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(Pipeline(stages), orch, repo, staging_dir=str(tmp_path / "uploads"))

    result = await engine.ingest_streams(
        [("report.pdf", io.BytesIO(b"%PDF-fake-bytes"))], parser="pdfplumber"
    )
    assert result.queued == 1

    meta = enq.jobs[0].item.metadata
    staged = meta["source_path"]
    assert staged.endswith(".pdf")  # extension preserved so the loader dispatches
    assert open(staged, "rb").read() == b"%PDF-fake-bytes"  # streamed to disk
    assert meta["source_type"] == "pdf"
    assert meta["parser"] == "pdfplumber"


async def test_ingest_streams_requires_staging_dir(repo):
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), InMemoryJobQueue(), repo, create_sink_registry())
    engine = IngestionEngine(Pipeline(stages), orch, repo)  # no staging_dir
    with pytest.raises(RuntimeError, match="not configured"):
        await engine.ingest_streams([("x.txt", io.BytesIO(b"hi"))])
