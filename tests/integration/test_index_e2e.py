"""End-to-end: a document through the whole DAG, persisted into the §8 sqlite-vec/FTS5 index.

InMemoryJobQueue + SqliteIndexStore (as the orchestrator's chunk_store) + the repo for
job_status + a fake 3-d embedder. Proves ingestion produces a queryable retrieval index.
"""

import sqlite_vec

from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.models import PipelineItem
from app.domains.base.status import DocumentStatusReader
from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.queue import InMemoryJobQueue
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

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": "fp-fake"}


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


async def test_ingest_builds_queryable_index(repo, tmp_path):
    queue = InMemoryJobQueue()
    index = SqliteIndexStore(str(tmp_path / "index.db"), embedding_dim=3).connect()
    index.write_index_meta(_FakeEmbedder())

    orch = PipelineOrchestrator(
        PipelineDAG(_stages()), queue, repo, create_sink_registry(), chunk_store=index
    )
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)

    await orch.ingest_documents(
        [PipelineItem(content="Hello world. " * 10, metadata={"source_id": "s1"})]
    )
    await queue.run()  # drain the whole DAG into the index

    # The §8 index is populated and queryable.
    chunk_count, embedding_count = index.counts("s1")
    assert chunk_count >= 2 and embedding_count == chunk_count

    q = sqlite_vec.serialize_float32([13.0, 1.0, 0.0])  # len("Hello world. ")=13
    knn = index.conn.execute(
        "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (q,),
    ).fetchall()
    assert len(knn) == 1
    fts = index.conn.execute(
        "SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH 'hello'"
    ).fetchone()[0]
    assert fts >= 1

    # The document row landed with default license metadata.
    lic = index.conn.execute(
        "SELECT license_class FROM documents WHERE document_id='s1'"
    ).fetchone()[0]
    assert lic == "public_domain"

    # job_status still recorded in the repository (not the index).
    jobs = await repo.document_jobs("s1")
    assert jobs and all(j["status"] == "completed" for j in jobs)

    # Status in retrieval mode: the reader composes repo job_status with index data facts.
    reader = DocumentStatusReader(repo, index)  # jobs from repo, facts from the §8 index
    status = await reader.document_status("s1", verbose=True)
    assert status["status"] == "complete"
    assert status["chunk_count"] == chunk_count
    assert status["embedding_count"] == embedding_count
    assert isinstance(status["jobs"], list) and status["jobs"]
    index.close()
