"""IngestionService facade on real InMemoryJobQueue + SQLite wiring (no API, no Postgres).

Uses a fake 3-d encoder so no sentence-transformers is needed (matches the e2e test).
"""

from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.pipeline import Pipeline
from app.domains.ingestion.queue import InMemoryJobQueue
from app.domains.ingestion.result_sink import create_sink_registry
from app.domains.ingestion.service import IngestionService
from app.domains.ingestion.stages.chunk import ChunkStage
from app.domains.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from app.domains.ingestion.stages.embed import EmbedStage
from app.domains.ingestion.stages.enrich import EnrichMetadataStage
from app.domains.ingestion.stages.load_parse import LoadAndParseStage
from app.domains.ingestion.worker import IngestionWorker


class _FakeEncoder:
    def encode(self, texts, convert_to_tensor=False):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeEmbedStage(EmbedStage):
    def _get_model(self):
        return _FakeEncoder()


def _stages():
    return [
        LoadAndParseStage(),
        CleanAndNormalizeStage(),
        ChunkStage(chunk_size=30, overlap=5),
        EnrichMetadataStage(),
        FakeEmbedStage(model_batch_size=2),
    ]


def _wire(repo):
    """Build the producer+consumer wiring around one repo; return (service, queue)."""
    queue = InMemoryJobQueue()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), queue, repo, create_sink_registry())
    service = IngestionService(Pipeline(stages), orch, repo)
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)
    return service, queue


async def test_ingest_from_content_queues_then_completes(repo):
    service, queue = _wire(repo)

    result = await service.ingest_from_content(
        [{"content": "Hello world. " * 10, "source_id": "s1"}]
    )
    assert result["documents_queued"] == 1
    assert result["documents"] == [{"document_id": "s1", "status": "queued"}]
    # Queued but not yet processed -> pending (no persisted data).
    assert (await service.get_document_status("s1"))["status"] == "pending"

    await queue.run()  # drain the whole DAG

    status = await service.get_document_status("s1")
    assert status["status"] == "complete"
    assert status["chunk_count"] >= 1
    assert status["embedding_count"] == status["chunk_count"]
    assert "jobs" not in status  # not verbose


async def test_unknown_document_status_is_none(repo):
    service, _ = _wire(repo)
    assert await service.get_document_status("does-not-exist") is None


async def test_verbose_adds_jobs_breakdown(repo):
    service, queue = _wire(repo)
    await service.ingest_from_content(
        [{"content": "alpha beta gamma. " * 8, "source_id": "s1"}]
    )
    await queue.run()

    status = await service.get_document_status("s1", verbose=True)
    assert isinstance(status["jobs"], list) and status["jobs"]
    assert all(j["status"] == "completed" for j in status["jobs"])


async def test_missing_source_id_gets_assigned(repo):
    service, _ = _wire(repo)
    result = await service.ingest_from_content([{"content": "x"}])
    assert result["documents_queued"] == 1
    assert result["documents"][0]["document_id"]  # a uuid was assigned
