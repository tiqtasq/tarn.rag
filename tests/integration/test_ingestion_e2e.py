"""End-to-end: a document through the whole pipeline on InMemoryJobQueue + SQLite.

No Postgres, no pgQueuer, no embedding model — the embed stage uses a fake encoder.
"""

from sqlalchemy import func, select

from tarnrag.storage.models import PipelineItem
from tarnrag.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.queue import InMemoryJobQueue
from tarnrag.ingestion.result_sink import create_sink_registry
from tarnrag.ingestion.stages.chunk import ChunkStage
from tarnrag.ingestion.stages.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.stages.embed import EmbedStage
from tarnrag.ingestion.stages.enrich import EnrichMetadataStage
from tarnrag.ingestion.stages.load_parse import LoadAndParseStage
from tarnrag.ingestion.worker import IngestionWorker


class _FakeEmbedder:
    """Deterministic 3-d encoder — no sentence-transformers needed."""

    def embed_passages(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeEmbedStage(EmbedStage):
    """EmbedStage wired with the fake encoder (reconstructed by the worker from config)."""

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


async def test_full_document_ingest(repo):
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(PipelineDAG(_stages()), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)  # pure handler; gets stages from the DAG
    queue.set_handler(worker.handle_batch)  # composition-root wiring

    item = PipelineItem(content="Hello world. " * 10, metadata={"source_id": "s1"})
    [doc_id] = await orch.ingest_documents([item])
    assert doc_id == "s1"

    await queue.run()  # drain the whole DAG, including chunk fan-out

    # Document fully ingested: doc + N chunks + N embeddings, no failures.
    status = await repo.document_status("s1")
    assert status["status"] == "complete"
    assert status["chunk_count"] >= 2
    assert status["embedding_count"] == status["chunk_count"]
    assert queue.dead_letters == []

    # Chunks were enriched (char_count) and their embeddings are searchable.
    results = await repo.vector_search([10.0, 1.0, 0.0], k=status["chunk_count"])
    assert len(results) == status["chunk_count"]
    assert all("char_count" in chunk.metadata for chunk, _ in results)

    # Every job for the document ended 'completed' (none failed/stuck).
    jobs = await repo.document_jobs("s1")
    assert jobs and all(j["status"] == "completed" for j in jobs)


async def test_reingest_is_idempotent(repo):
    """Re-ingesting the same source_id replaces chunks/embeddings, not duplicates."""
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(PipelineDAG(_stages()), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)

    item = PipelineItem(content="alpha beta gamma. " * 8, metadata={"source_id": "s1"})
    await orch.ingest_documents([item])
    await queue.run()
    first = await repo.document_status("s1")

    await orch.ingest_documents(
        [PipelineItem(content="alpha beta gamma. " * 8, metadata={"source_id": "s1"})]
    )
    await queue.run()
    second = await repo.document_status("s1")

    assert second["status"] == "complete"
    # Counts didn't double — the document was upserted and its chunks replaced.
    assert second["chunk_count"] == first["chunk_count"]
    assert second["embedding_count"] == first["embedding_count"]

    # Prove no orphans: the raw embeddings/chunks tables hold exactly the live rows
    # (so the FK ON DELETE CASCADE actually fired when the old chunks were dropped).
    async with repo.engine.connect() as conn:
        raw_chunks = (await conn.execute(select(func.count()).select_from(repo.chunks))).scalar_one()
        raw_embs = (await conn.execute(select(func.count()).select_from(repo.embeddings))).scalar_one()
    assert raw_chunks == second["chunk_count"]
    assert raw_embs == second["embedding_count"]
