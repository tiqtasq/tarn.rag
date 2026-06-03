"""End-to-end API flow over httpx ASGITransport, with DI overridden to InMemory + SQLite.

No Postgres, no pgQueuer, no embedding model. The app's real lifespan never runs (ASGI
transport doesn't emit lifespan events); we override ``get_ingestion_service`` to inject a
test wiring and drive the same queue to drain the DAG.
"""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.v1.dependencies import get_ingestion_service
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
from app.main import create_app


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


@pytest_asyncio.fixture
async def api(repo):
    """Yield (client, queue): an httpx client bound to the app (with the service dependency
    overridden to InMemory+SQLite) plus the queue so tests can drain the DAG."""
    queue = InMemoryJobQueue()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), queue, repo, create_sink_registry())
    service = IngestionService(Pipeline(stages), orch, repo)
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)

    app = create_app()
    app.dependency_overrides[get_ingestion_service] = lambda: service
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, queue


async def test_ingest_content_returns_document_ids(api):
    client, _ = api
    resp = await client.post(
        "/v1/ingest/content",
        json={"documents": [{"content": "Hello world. " * 10, "source_id": "s1"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents_queued"] == 1
    assert body["documents"][0] == {"document_id": "s1", "status": "queued"}


async def test_status_lifecycle_pending_then_complete(api):
    client, queue = api
    await client.post(
        "/v1/ingest/content",
        json={"documents": [{"content": "Hello world. " * 10, "source_id": "s1"}]},
    )

    before = await client.get("/v1/ingest/documents/s1/status")
    assert before.status_code == 200
    assert before.json()["status"] == "pending"

    await queue.run()  # drain the whole DAG

    after = (await client.get("/v1/ingest/documents/s1/status")).json()
    assert after["status"] == "complete"
    assert after["embedding_count"] == after["chunk_count"] > 0
    assert after["jobs"] is None  # not verbose


async def test_status_404_for_unknown_document(api):
    client, _ = api
    resp = await client.get("/v1/ingest/documents/ghost/status")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Document not found"


async def test_verbose_includes_jobs(api):
    client, queue = api
    await client.post(
        "/v1/ingest/content",
        json={"documents": [{"content": "alpha beta gamma. " * 8, "source_id": "s1"}]},
    )
    await queue.run()

    resp = await client.get(
        "/v1/ingest/documents/s1/status", params={"verbose": "true"}
    )
    jobs = resp.json()["jobs"]
    assert isinstance(jobs, list) and jobs


async def test_ingest_from_paths_queues_one_job_per_path(api):
    client, _ = api  # don't drain: LoadAndParse would read the (absent) files
    resp = await client.post(
        "/v1/ingest/", json={"file_paths": ["/tmp/a.txt", "/tmp/b.pdf"]}
    )
    assert resp.status_code == 200
    assert resp.json()["documents_queued"] == 2


async def test_known_parser_is_accepted(api):
    client, _ = api
    resp = await client.post(
        "/v1/ingest/content",
        json={"documents": [{"content": "x", "source_id": "s1"}], "parser": "pdfplumber"},
    )
    assert resp.status_code == 200


async def test_unknown_parser_is_rejected_422(api):
    client, _ = api
    resp = await client.post(
        "/v1/ingest/content",
        json={"documents": [{"content": "x", "source_id": "s1"}], "parser": "bogus"},
    )
    assert resp.status_code == 422  # rejected at the edge, nothing queued
