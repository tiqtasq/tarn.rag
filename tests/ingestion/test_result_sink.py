from tarnrag.contracts import Chunk, Document, Embedding, PipelineItem
from tarnrag.ingestion.result_sink import (
    ChunkMetadataResultSink,
    ChunkResultSink,
    DocumentResultSink,
    EmbeddingResultSink,
    PassthroughSink,
    create_sink_registry,
)


async def _finalize(sink, results):
    sink.submit(results)
    sink.close()
    return await sink.finalize()


async def test_document_sink_persists_and_threads_doc_id(repo):
    item = PipelineItem(content="doc text", metadata={"source_id": "s1"})
    outcome = await _finalize(DocumentResultSink(repo), [item])
    assert outcome.persisted
    doc_id = item.metadata["doc_id"]  # threaded back onto the item
    stored = await repo.get_document(doc_id)
    assert stored.content == "doc text"


async def test_chunk_sink_persists_and_threads_chunk_id(repo):
    doc_id = await repo.store_document(Document(content="d", metadata={"source_id": "s1"}))
    items = [
        PipelineItem(
            content=f"c{i}",
            metadata={"source_id": "s1", "doc_id": doc_id, "chunk_index": i, "total_chunks": 2},
        )
        for i in range(2)
    ]
    assert (await _finalize(ChunkResultSink(repo), items)).persisted
    assert all("chunk_id" in it.metadata for it in items)
    chunks = await repo.get_chunks_by_document(doc_id)
    assert [c.content for c in chunks] == ["c0", "c1"]
    assert [c.id for c in chunks] == [it.metadata["chunk_id"] for it in items]


async def test_enrich_sink_is_noop(repo):
    # §8 has no chunk metadata column — the enrich sink finalizes successfully but stores nothing.
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [Chunk(parent_doc_id="", content="c", chunk_index=0, total_chunks=1, metadata={})],
    )
    item = PipelineItem(content="c", metadata={"chunk_id": cid, "char_count": 1, "word_count": 1})
    assert (await _finalize(ChunkMetadataResultSink(repo), [item])).persisted
    assert "char_count" not in (await repo.get_chunk(cid)).metadata


async def test_embedding_sink_persists(repo):
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [Chunk(parent_doc_id="", content="c", chunk_index=0, total_chunks=1, metadata={})],
    )
    emb = Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0], model="m", dimension=3)
    assert (await _finalize(EmbeddingResultSink(repo), [emb])).persisted
    cands = await repo.dense_knn([1.0, 0.0, 0.0], k=1)
    assert cands[0].chunk_id == cid


async def test_passthrough_persists_nothing(repo):
    outcome = await _finalize(PassthroughSink(repo), [PipelineItem(content="x", metadata={})])
    assert outcome.persisted


async def test_finalize_reports_failure(repo):
    # A chunk referencing a missing document -> FK violation -> finalize reports not-persisted.
    item = PipelineItem(
        content="c", metadata={"doc_id": "missing", "chunk_index": 0, "total_chunks": 1}
    )
    outcome = await _finalize(ChunkResultSink(repo), [item])
    assert not outcome.persisted
    assert outcome.detail  # carries the DB error detail


def test_registry_covers_all_stages():
    registry = create_sink_registry()
    assert set(registry) == {
        "LoadAndParse",
        "CleanAndNormalize",
        "Chunk",
        "EnrichMetadata",
        "Embed",
    }
