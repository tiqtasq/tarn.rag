"""RetrievalEngine (dense-only) over a §8 index built with a fake embedder."""

import pytest

from tarnrag.storage.index_store import SqliteIndexStore
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
        return {
            "embedding_dim": "3",
            "embedding_config_fingerprint": FINGERPRINT,
        }


async def _index(tmp_path):
    store = SqliteIndexStore(str(tmp_path / "index.db"), embedding_dim=3).connect()
    store.write_index_meta(_FakeEmbedder())
    await store.store_document(Document(content="d", metadata={"source_id": "s1", "title": "T"}))
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="storage tank inspection",
              chunk_index=0, total_chunks=2, metadata={"locator": "§6.4"}),
        Chunk(parent_doc_id="s1", content="quokka marsupial",
              chunk_index=1, total_chunks=2, metadata={}),
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="f", dimension=3),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0], model="f", dimension=3),
    ])
    return store, cids


async def test_open_validates_fingerprint(tmp_path):
    store, _ = await _index(tmp_path)
    # Matching fingerprint -> opens.
    RetrievalEngine.open(store, _FakeEmbedder())
    # Mismatched fingerprint -> refuses.
    with pytest.raises(RetrievalError, match="fingerprint mismatch"):
        RetrievalEngine.open(store, _FakeEmbedder(fingerprint="other"))
    store.close()


async def test_search_ranks_nearest_with_provenance(tmp_path):
    store, cids = await _index(tmp_path)
    engine = RetrievalEngine.open(store, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))

    results = engine.search(Query(text="tank inspection", top_k=2))
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
    store.close()


async def test_top_k_truncates(tmp_path):
    store, _ = await _index(tmp_path)
    engine = RetrievalEngine.open(store, _FakeEmbedder())
    assert len(engine.search(Query(text="x", top_k=1))) == 1
    store.close()


async def test_empty_index_returns_empty(tmp_path):
    store = SqliteIndexStore(str(tmp_path / "empty.db"), embedding_dim=3).connect()
    store.write_index_meta(_FakeEmbedder())
    engine = RetrievalEngine.open(store, _FakeEmbedder())
    assert engine.search(Query(text="x")) == []
    store.close()


async def test_search_text_convenience(tmp_path):
    store, cids = await _index(tmp_path)
    engine = RetrievalEngine.open(store, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = engine.search_text("tank inspection", top_k=2)
    assert [r.chunk_id for r in results] == cids
    store.close()


async def test_async_variants_match_sync(tmp_path):
    store, cids = await _index(tmp_path)
    engine = RetrievalEngine.open(store, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    assert [r.chunk_id for r in await engine.asearch(Query(text="tank", top_k=2))] == cids
    assert [r.chunk_id for r in await engine.asearch_text("tank", top_k=2)] == cids
    store.close()


async def test_open_refuses_unbuilt_index(tmp_path):
    store = SqliteIndexStore(str(tmp_path / "fresh.db"), embedding_dim=3).connect()
    # No write_index_meta -> the index has not been built yet.
    with pytest.raises(RetrievalError, match="not been built"):
        RetrievalEngine.open(store, _FakeEmbedder())
    store.close()
