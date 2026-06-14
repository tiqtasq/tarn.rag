"""SqliteIndexStore — the §8 retrieval index write side, queried with sqlite-vec + FTS5.

Uses a fake embedder (just index_meta) and hand-set vectors — no model download.
"""

import sqlite3

import pytest
import sqlite_vec

from tarnrag.storage.index_store import SqliteIndexStore
from tarnrag.storage.models import Chunk, Document, Embedding


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


async def test_delete_document_removes_all_rows(tmp_path):
    store = _store(tmp_path)
    await store.store_document(Document(content="d", metadata={"source_id": "s1", "content_hash": "h1"}))
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="alpha", chunk_index=0, total_chunks=1, metadata={})
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3)
    ])
    assert store.counts("s1") == (1, 1)

    assert await store.delete_document("s1") is True
    assert store.counts("s1") == (0, 0)
    assert store.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM fts_chunks").fetchone()[0] == 0
    assert await store.delete_document("s1") is False  # already gone — idempotent
    store.close()


async def test_list_documents_with_counts(tmp_path):
    store = _store(tmp_path)
    await store.store_document(Document(content="d", metadata={"source_id": "s1", "content_hash": "h1"}))
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="a", chunk_index=0, total_chunks=2, metadata={}),
        Chunk(parent_doc_id="s1", content="b", chunk_index=1, total_chunks=2, metadata={}),
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3),
    ])
    # A second document with no chunks/embeddings.
    await store.store_document(Document(content="e", metadata={"source_id": "s2", "content_hash": "h2"}))

    docs = {d["document_id"]: d for d in await store.list_documents()}
    assert set(docs) == {"s1", "s2"}
    assert docs["s1"] == {
        "document_id": "s1", "content_hash": "h1", "chunk_count": 2, "embedding_count": 1,
    }
    assert docs["s2"]["content_hash"] == "h2"
    assert docs["s2"]["chunk_count"] == 0 and docs["s2"]["embedding_count"] == 0
    store.close()


async def test_delete_document_rolls_back_on_failure(tmp_path):
    """If a delete fails mid-way, the whole transaction rolls back — no partial removal."""
    store = _store(tmp_path)
    await store.store_document(Document(content="d", metadata={"source_id": "s1"}))
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="a", chunk_index=0, total_chunks=1, metadata={})
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3)
    ])
    assert store.counts("s1") == (1, 1)

    # Wrap the connection so the final 'DELETE FROM documents' raises mid-transaction; the
    # earlier vec/fts/method/chunks deletes have already run on the real connection.
    real = store._conn

    class _FailDocs:
        def execute(self, sql, *args):
            if sql.startswith("DELETE FROM documents"):
                raise sqlite3.OperationalError("boom")
            return real.execute(sql, *args)

        def __enter__(self):
            return real.__enter__()

        def __exit__(self, *exc):
            return real.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(real, name)

    store._conn = _FailDocs()
    with pytest.raises(sqlite3.OperationalError):
        await store.delete_document("s1")
    store._conn = real  # restore

    # The transaction rolled back: everything is still there, nothing partially deleted.
    assert store.counts("s1") == (1, 1)
    assert store.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 1
    assert store.conn.execute("SELECT count(*) FROM fts_chunks").fetchone()[0] == 1
    store.close()


async def test_store_document_reingest_rolls_back_on_failure(tmp_path):
    """Re-storing deletes old chunks then upserts the doc; if the upsert fails, the deletes must
    roll back — otherwise a failed re-ingest would lose the old data."""
    store = _store(tmp_path)
    await store.store_document(Document(content="d", metadata={"source_id": "s1"}))
    cids = await store.store_chunks([
        Chunk(parent_doc_id="s1", content="a", chunk_index=0, total_chunks=1, metadata={})
    ])
    await store.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0], model="fake", dimension=3)
    ])
    assert store.counts("s1") == (1, 1)

    real = store._conn

    class _FailUpsert:
        def execute(self, sql, *args):
            if sql.startswith("INSERT OR REPLACE INTO documents"):
                raise sqlite3.OperationalError("boom")
            return real.execute(sql, *args)

        def __enter__(self):
            return real.__enter__()

        def __exit__(self, *exc):
            return real.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(real, name)

    store._conn = _FailUpsert()
    with pytest.raises(sqlite3.OperationalError):
        await store.store_document(Document(content="d2", metadata={"source_id": "s1"}))
    store._conn = real  # restore

    # The chunk-deletes rolled back with the failed upsert: old chunk + vector survive intact.
    assert store.counts("s1") == (1, 1)
    store.close()
