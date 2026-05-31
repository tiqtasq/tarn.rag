import pytest

from app.core.exceptions import ChunkNotFoundError
from app.domains.base.models import Chunk, Document, Embedding


def _chunk(content, index, total, source_id="s1", **meta):
    return Chunk(
        parent_doc_id="",  # overwritten by store_document_with_chunks
        content=content,
        chunk_index=index,
        total_chunks=total,
        metadata={"source_id": source_id, **meta},
    )


async def test_store_and_get_document(repo):
    doc_id = await repo.store_document(
        Document(content="hello", metadata={"source_id": "s1"})
    )
    got = await repo.get_document(doc_id)
    assert got is not None
    assert got.content == "hello"
    assert got.metadata["source_id"] == "s1"


async def test_document_with_chunks_and_idempotency(repo):
    doc_id, ids = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}),
        [_chunk("c0", 0, 2), _chunk("c1", 1, 2)],
    )
    assert len(ids) == 2
    assert len(await repo.get_chunks_by_document(doc_id)) == 2

    # Re-ingesting the same source_id upserts the document (same id) and replaces chunks.
    doc_id2, ids2 = await repo.store_document_with_chunks(
        Document(content="updated", metadata={"source_id": "s1"}),
        [_chunk("only", 0, 1)],
    )
    assert doc_id2 == doc_id
    remaining = await repo.get_chunks_by_document(doc_id)
    assert [c.content for c in remaining] == ["only"]
    assert (await repo.get_document(doc_id)).content == "updated"


async def test_embeddings_and_vector_search(repo):
    doc_id, (cid_a, cid_b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 2), _chunk("b", 1, 2)],
    )
    await repo.store_embeddings(
        [
            Embedding(chunk_id=cid_a, vector=[1.0, 0.0, 0.0], model="m", dimension=3),
            Embedding(chunk_id=cid_b, vector=[0.0, 1.0, 0.0], model="m", dimension=3),
        ]
    )
    results = await repo.vector_search([0.9, 0.1, 0.0], k=2)
    assert len(results) == 2
    (top_chunk, top_sim), (_, second_sim) = results
    assert top_chunk.id == cid_a  # closest to [1,0,0]
    assert top_sim > second_sim


async def test_update_chunk_metadata(repo):
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1, k=1)],
    )
    await repo.update_chunk_metadata(cid, {"k": 2, "extra": "x"})
    ch = await repo.get_chunk(cid)
    assert ch.metadata["k"] == 2 and ch.metadata["extra"] == "x"

    with pytest.raises(ChunkNotFoundError):
        await repo.update_chunk_metadata("missing", {"a": 1})


async def test_document_status_and_jobs(repo):
    assert await repo.document_status("unknown") is None

    # A queued job with no document yet -> pending.
    await repo.record_job("s1", "j1", "LoadAndParse", "queued")
    assert (await repo.document_status("s1"))["status"] == "pending"

    # Document + chunk + embedding -> complete.
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1)],
    )
    await repo.store_embeddings(
        [Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0], model="m", dimension=3)]
    )
    assert await repo.document_status("s1") == {
        "document_id": "s1",
        "status": "complete",
        "chunk_count": 1,
        "embedding_count": 1,
    }

    # A failed job rolls the document status up to 'failed'.
    await repo.record_job("s1", "j2", "Embed", "failed", "boom")
    assert (await repo.document_status("s1"))["status"] == "failed"

    jobs = await repo.document_jobs("s1")
    assert len(jobs) == 2
    assert any(j["status"] == "failed" and j["error"] == "boom" for j in jobs)
