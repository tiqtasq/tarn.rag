"""RetrievalEngine (dense-only) over the §8 repository, built with a fake embedder."""

import pytest

from tarnrag.storage.index_meta import build_index_meta
from tarnrag.storage.models import Chunk, Document, Embedding
from tarnrag.retrieval import Query, RetrievalEngine, RetrievalError

FINGERPRINT = "fp-123"


class _FakeEmbedder:
    """Embeds a query to a fixed 3-d vector; fingerprint matches the index by default."""

    def __init__(self, query_vec=(1.0, 0.0, 0.0), fingerprint=FINGERPRINT):
        self._vec = list(query_vec)
        self._fp = fingerprint

    def embed_query(self, text):
        return self._vec

    def config_fingerprint(self):
        return self._fp

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": FINGERPRINT}


async def _index(repo):
    """Stamp index_meta and ingest two chunks (+ embeddings) into the repo; return the chunk ids."""
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))
    _, cids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1", "title": "T"}),
        [
            Chunk(parent_doc_id="s1", content="storage tank inspection",
                  chunk_index=0, total_chunks=2, metadata={"locator": "§6.4"}),
            Chunk(parent_doc_id="s1", content="quokka marsupial",
                  chunk_index=1, total_chunks=2, metadata={}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0], model="f", dimension=3),
    ])
    return cids


async def test_open_validates_fingerprint(repo):
    await _index(repo)
    # Matching fingerprint -> opens.
    await RetrievalEngine.open(repo, _FakeEmbedder())
    # Mismatched fingerprint -> refuses.
    with pytest.raises(RetrievalError, match="fingerprint mismatch"):
        await RetrievalEngine.open(repo, _FakeEmbedder(fingerprint="other"))


async def test_search_ranks_nearest_with_provenance(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))

    results = await engine.search(Query(text="tank inspection", top_k=2))
    assert [r.chunk_id for r in results] == cids  # chunk 0 ([1,0,0]) nearest
    top = results[0]
    assert top.text == "storage tank inspection"
    assert top.document_id == "s1"
    assert top.source_kind == "document"
    assert top.license_class == "public_domain"
    assert top.locator == "§6.4"
    assert top.methods == []  # method_chunks empty for now
    assert top.score >= results[1].score  # ranked by score desc
    assert "dense" in top.component_scores


async def test_top_k_truncates(repo):
    await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert len(await engine.search(Query(text="x", top_k=1))) == 1


async def test_empty_index_returns_empty(repo):
    await repo.write_index_meta(build_index_meta(_FakeEmbedder()))  # built, but no chunks
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert await engine.search(Query(text="x")) == []


async def test_search_text_convenience(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = await engine.search_text("tank inspection", top_k=2)
    assert [r.chunk_id for r in results] == cids


async def test_open_refuses_unbuilt_index(repo):
    # No write_index_meta -> the index has not been built yet.
    with pytest.raises(RetrievalError, match="not been built"):
        await RetrievalEngine.open(repo, _FakeEmbedder())
