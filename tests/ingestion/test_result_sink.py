import hashlib

from tarnrag.contracts import Chunk, ChunkProvenance, Document, Embedding, PipelineItem
from tarnrag.ingestion.engine.result_sink import (
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


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


async def test_chunk_sink_threads_provenance_and_resolves_the_tree(repo):
    doc_id = await repo.store_document(Document(content="d", metadata={"source_id": "s1"}))
    parent = PipelineItem(
        content="Safety",
        metadata={"source_id": "s1", "doc_id": doc_id, "chunk_index": 0, "parent_ordinal": None},
        provenance=ChunkProvenance(content_hash=_h("Safety"), header_path=["Safety"], level=1),
    )
    leaf = PipelineItem(
        content="Wear PPE.",
        metadata={"source_id": "s1", "doc_id": doc_id, "chunk_index": 1, "parent_ordinal": 0},
        provenance=ChunkProvenance(content_hash=_h("Wear PPE."), header_path=["Safety"], level=0),
    )
    assert (await _finalize(ChunkResultSink(repo), [parent, leaf])).persisted
    stored_parent, stored_leaf = await repo.get_chunks_by_document(doc_id)
    assert stored_parent.provenance.level == 1 and stored_parent.provenance.header_path == ["Safety"]
    assert stored_leaf.provenance.parent_chunk_id == stored_parent.id  # sink + repo resolved the tree


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
        "Enrich",
        "CleanAndNormalize",
        "Chunk",
        "Embed",
    }
