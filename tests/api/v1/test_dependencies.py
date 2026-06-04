"""Composition builders: index-mode vs classic-mode wiring of the data store + status facts."""

from app.api.v1.dependencies import build_service
from app.core.config import Settings
from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.models import Chunk, Document, Embedding
from app.domains.ingestion.queue import InMemoryJobQueue


class _FakeEmbedder:
    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": "fp"}


def _settings(tmp_path):
    return Settings(
        QUEUE_DB_URL="postgresql://x",  # not connected here
        DOCUMENT_DB_URL=f"sqlite:///{tmp_path}/docs.db",
        MODEL_DIR=str(tmp_path / "model"),
        INDEX_DB_PATH=str(tmp_path / "index.db"),
        EMBEDDING_DIMENSION=3,
    )


async def test_index_mode_wires_index_as_store_and_status_facts(repo, tmp_path):
    index = SqliteIndexStore(str(tmp_path / "index.db"), embedding_dim=3).connect()
    index.write_index_meta(_FakeEmbedder())
    service = build_service(_settings(tmp_path), InMemoryJobQueue(), repo, index_store=index)

    # Sinks persist into the index; job_status stays on the repo.
    assert service.orchestrator.chunk_store is index

    # Status composes repo job_status with index data facts.
    await repo.record_job("s1", "j1", "Embed", "completed")
    await index.store_document(Document(content="d", metadata={"source_id": "s1"}))
    [cid] = await index.store_chunks(
        [Chunk(parent_doc_id="s1", content="a", chunk_index=0, total_chunks=1, metadata={})]
    )
    await index.store_embeddings(
        [Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0], model="f", dimension=3)]
    )
    status = await service.get_document_status("s1", verbose=True)
    assert status["status"] == "complete"
    assert status["chunk_count"] == 1 and status["embedding_count"] == 1
    assert status["jobs"]
    index.close()


async def test_classic_mode_defaults_to_repo(repo, tmp_path):
    # No index_store -> data + status facts both come from the repository (unchanged path).
    service = build_service(_settings(tmp_path), InMemoryJobQueue(), repo)
    assert service.orchestrator.chunk_store is repo
