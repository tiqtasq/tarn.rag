"""Batched dispatch + bulk-persist must be *equivalent* to one-doc-at-a-time ingest — same documents, same
chunks, same embeddings — only fewer transactions. Driven by the model-free ``hash`` embedder (offline)."""

from tarnrag.core.engine.config import DatabaseSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.storage.repository import DocumentRepository


async def _ingest(db_path: str, max_batch: int) -> dict[str, tuple[int, int]]:
    """Ingest a fixed doc set at the given dispatch batch size; return ``{doc_id: (chunks, embeddings)}``."""
    settings = Settings(_env_file=None, ID_POLICY="caller", embedding={"provider": "hash"})
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    repo = await DocumentRepository.create(DatabaseSettings(document_url=f"sqlite:///{db_path}"), settings.EMBEDDING_DIMENSION)
    await IngestionEngine.ensure_index_meta(repo, embedder)
    ing = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
    ing._queue.max_batch_size = max_batch
    docs = [
        {"source_id": f"d{i}", "content": f"Document {i}. Section one. Section two. " * 8, "title": f"d{i}", "source_type": "text"}
        for i in range(12)
    ]
    await ing.ingest_content(docs)
    summaries = {s.document_id: (s.chunk_count, s.embedding_count) for s in await ing.list_documents()}
    await repo.disconnect()
    return summaries


async def test_batched_bulk_ingest_matches_per_doc(tmp_path):
    per_doc = await _ingest(str(tmp_path / "solo.db"), max_batch=1)  # one doc per dispatch (old behavior)
    batched = await _ingest(str(tmp_path / "batch.db"), max_batch=64)  # whole batch in one dispatch + bulk-persist
    assert len(per_doc) == 12
    assert per_doc == batched  # identical index: same docs, same chunk + embedding counts
