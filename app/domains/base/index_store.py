"""The §8 retrieval index — a single SQLite file with sqlite-vec (dense KNN) + FTS5 (sparse).

This is the **write** side, fed by the ingestion ResultSinks through the ``ChunkStore`` seam.
The schema is `doc/ModusQ_RetrievalSubsystemSpec.md` §8.1; the read side (`dense_knn`,
`sparse_bm25`, `hydrate`) is added with the retrieval engine — same class, same file.

Domain fields (license_class, methods) are **deferred**: populated with safe defaults so the
index is well-formed. The embedding pipeline identity is recorded in ``index_meta`` (incl. the
``embedding_config_fingerprint`` retrieval validates at ``open()``).

Sync ``sqlite3`` is used deliberately (loadable extensions are simplest there, and the future
C++ port is sync too); the async ``store_*`` methods wrap small in-process writes.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from typing import Any

import sqlite_vec

from app.domains.base.chunk_store import ChunkStore
from app.domains.base.models import Chunk, Document, Embedding
from app.domains.base.status import DocumentFacts, DocumentFactsSource

SCHEMA_VERSION = "1"
INGESTION_VERSION = "0.1.0"
FTS_TOKENIZER = "unicode61"


class SqliteIndexStore(ChunkStore, DocumentFactsSource):
    def __init__(
        self,
        db_path: str,
        embedding_dim: int,
        *,
        default_license_class: str = "public_domain",
        default_ai_grounding: int = 1,
    ):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.default_license_class = default_license_class
        self.default_ai_grounding = default_ai_grounding
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteIndexStore.connect() must be called first")
        return self._conn

    def connect(self) -> SqliteIndexStore:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys=ON")
        # WAL so API readers (status facts) don't block the worker writer on the same file.
        conn.execute("PRAGMA journal_mode=WAL")
        self._conn = conn
        self._create_schema()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------------- schema + meta ----------------

    def _create_schema(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
              document_id   TEXT PRIMARY KEY,
              title         TEXT,
              source_kind   TEXT NOT NULL,
              standard_id   TEXT,
              doc_version   TEXT,
              license_class TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id             TEXT PRIMARY KEY,
              document_id          TEXT NOT NULL REFERENCES documents(document_id),
              ordinal              INTEGER NOT NULL,
              text                 TEXT NOT NULL,
              locator              TEXT,
              license_class        TEXT NOT NULL,
              ai_grounding_allowed INTEGER NOT NULL,
              available            INTEGER NOT NULL DEFAULT 1,
              content_hash         TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS method_chunks (
              method_id      TEXT NOT NULL,
              method_version TEXT NOT NULL,
              chunk_id       TEXT NOT NULL REFERENCES chunks(chunk_id),
              PRIMARY KEY (method_id, method_version, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_method_chunks_chunk ON method_chunks(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_license ON chunks(license_class, available);
            """
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{self.embedding_dim}])"
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
            f"chunk_id UNINDEXED, text, tokenize='{FTS_TOKENIZER}')"
        )
        c.commit()

    def write_index_meta(self, embedder: Any, extra: dict[str, str] | None = None) -> None:
        """Record build + embedding-pipeline identity (once). ``embedder.embed_meta()`` supplies
        model/tokenizer/pooling/normalize/prefixes/dim + the ``embedding_config_fingerprint``."""
        meta = {
            "schema_version": SCHEMA_VERSION,
            "ingestion_version": INGESTION_VERSION,
            "built_at": str(int(time.time())),
            "fts_tokenizer": FTS_TOKENIZER,
            **embedder.embed_meta(),
            **(extra or {}),
        }
        self.conn.executemany(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in meta.items()],
        )
        self.conn.commit()

    def index_meta(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT key, value FROM index_meta").fetchall())

    # ---------------- ChunkStore (write side) ----------------

    async def store_document(self, doc: Document) -> str:
        md = doc.metadata or {}
        document_id = md.get("source_id") or str(uuid.uuid4())
        # Idempotent (re)store: drop the doc's existing chunks + their vec/fts/method rows
        # (virtual tables don't cascade), then upsert the document.
        old = [r[0] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE document_id=?", (document_id,)
        )]
        if old:
            marks = ",".join("?" * len(old))
            self.conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute(f"DELETE FROM method_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.conn.execute(
            "INSERT OR REPLACE INTO documents"
            "(document_id, title, source_kind, standard_id, doc_version, license_class)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                document_id,
                md.get("title"),
                md.get("source_kind") or "document",
                md.get("standard_id"),
                md.get("doc_version"),
                md.get("license_class") or self.default_license_class,
            ),
        )
        self.conn.commit()
        return document_id

    async def store_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []
        ids: list[str] = []
        crows: list[tuple] = []
        frows: list[tuple] = []
        for ch in chunks:
            cid = ch.id or str(uuid.uuid4())
            ids.append(cid)
            md = ch.metadata or {}
            crows.append((
                cid,
                ch.parent_doc_id,
                ch.chunk_index,
                ch.content,
                md.get("locator"),
                md.get("license_class") or self.default_license_class,
                int(md.get("ai_grounding_allowed", self.default_ai_grounding)),
                int(md.get("available", 1)),
                hashlib.sha256(ch.content.encode("utf-8")).hexdigest(),
            ))
            frows.append((cid, ch.content))
        self.conn.executemany(
            "INSERT INTO chunks(chunk_id, document_id, ordinal, text, locator, "
            "license_class, ai_grounding_allowed, available, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            crows,
        )
        self.conn.executemany(
            "INSERT INTO fts_chunks(chunk_id, text) VALUES (?, ?)", frows
        )
        self.conn.commit()
        return ids

    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        if not embeddings:
            return []
        rows = [
            (e.chunk_id, sqlite_vec.serialize_float32(e.vector)) for e in embeddings
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)", rows
        )
        self.conn.commit()
        return [e.chunk_id for e in embeddings]

    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        # §8 chunks has no metadata column; enrichment (char/word counts) isn't part of the
        # retrieval index. No-op by design.
        return None

    # ---------------- status support ----------------

    def counts(self, document_id: str) -> tuple[int, int]:
        """(chunk_count, embedding_count) for the document."""
        chunk_count = self.conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id=?", (document_id,)
        ).fetchone()[0]
        embedding_count = self.conn.execute(
            "SELECT count(*) FROM vec_chunks WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE document_id=?)",
            (document_id,),
        ).fetchone()[0]
        return chunk_count, embedding_count

    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Data facts from the index — feeds the ``DocumentStatusReader`` in retrieval mode."""
        present = self.conn.execute(
            "SELECT 1 FROM documents WHERE document_id=?", (document_id,)
        ).fetchone() is not None
        chunk_count, embedding_count = self.counts(document_id)
        return DocumentFacts(present, chunk_count, embedding_count)
