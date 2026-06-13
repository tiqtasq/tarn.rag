"""SqliteIndexStore — the §8 retrieval index write side, queried with sqlite-vec + FTS5.

Uses a fake embedder (just index_meta) and hand-set vectors — no model download.
"""

import sqlite_vec

from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.models import Chunk, Document, Embedding


class _FakeEmbedder:
    def embed_meta(self):
        return {
            "embedding_model_id": "fake",
            "embedding_model_revision": "",
            "embedding_dim": "3",
            "tokenizer_sha256": "deadbeef",
            "pooling": "mean",
            "normalize": "l2",
            "query_prefix": "",
            "passage_prefix": "",
            "max_length": "512",
            "embedding_config_fingerprint": "fp-123",
        }


def _store(tmp_path):
    store = SqliteIndexStore(str(tmp_path / "index.db"), embedding_dim=3).connect()
    store.write_index_meta(_FakeEmbedder())
    return store


def _knn(store, vec, k=2):
    q = sqlite_vec.serialize_float32(vec)
    return store.conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT ?",
        (q, k),
    ).fetchall()


def test_connect_creates_missing_parent_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    store = SqliteIndexStore(str(nested / "index.db"), embedding_dim=3).connect()
    assert nested.is_dir()  # SQLite won't create it; connect() does
    store.close()


def test_index_meta_records_schema_and_fingerprint(tmp_path):
    store = _store(tmp_path)
    meta = store.index_meta()
    assert meta["schema_version"] == "1"
    assert meta["fts_tokenizer"] == "unicode61"
    assert meta["embedding_config_fingerprint"] == "fp-123"
    assert meta["embedding_dim"] == "3"
    store.close()


async def test_roundtrip_knn_bm25_and_counts(tmp_path):
    store = _store(tmp_path)
    doc_id = await store.store_document(
        Document(content="d", metadata={"source_id": "s1", "title": "T"})
    )
    assert doc_id == "s1"

    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="storage tank inspection",
              chunk_index=0, total_chunks=2, metadata={"locator": "§1"}),
        Chunk(parent_doc_id="s1", content="quokka marsupial",
              chunk_index=1, total_chunks=2, metadata={}),
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0], model="fake", dimension=3),
    ])

    assert store.counts("s1") == (2, 2)

    # dense KNN: nearest to [1,0,0] is chunk 0
    assert _knn(store, [1.0, 0.0, 0.0])[0][0] == cids[0]
    # sparse BM25: 'tank' matches only chunk 0; provenance/locator persisted
    fts = store.conn.execute(
        "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH 'tank'"
    ).fetchall()
    assert fts == [(cids[0],)]
    loc = store.conn.execute(
        "SELECT locator, license_class, available FROM chunks WHERE chunk_id=?", (cids[0],)
    ).fetchone()
    assert loc == ("§1", "public_domain", 1)
    store.close()


async def test_reingest_replaces_chunks_no_orphans(tmp_path):
    store = _store(tmp_path)
    item = Document(content="d", metadata={"source_id": "s1"})
    await store.store_document(item)
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="alpha", chunk_index=0, total_chunks=1, metadata={})
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3)
    ])
    assert store.counts("s1") == (1, 1)

    # Re-store the document: old chunks (and their vec/fts rows) are dropped.
    await store.store_document(item)
    assert store.counts("s1") == (0, 0)
    raw_vec = store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    raw_fts = store.conn.execute("SELECT count(*) FROM fts_chunks").fetchone()[0]
    assert raw_vec == 0 and raw_fts == 0  # no orphaned vectors / fts rows
    store.close()
