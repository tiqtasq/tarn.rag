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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlite_vec

from tarnrag.core.config import IndexSettings
from tarnrag.storage.chunk_store import ChunkStore
from tarnrag.storage.models import Chunk, Document, Embedding
from tarnrag.storage.status import DocumentFacts, DocumentFactsSource

SCHEMA_VERSION = "1"
INGESTION_VERSION = "0.1.0"
FTS_TOKENIZER = "unicode61"


@dataclass(frozen=True)
class Candidate:
    """A ranked candidate from a retriever (rank is 1-based; raw_score is engine-specific:
    distance for dense KNN, bm25 for sparse)."""

    chunk_id: str
    rank: int
    raw_score: float


@dataclass(frozen=True)
class ChunkRecord:
    """A hydrated chunk: canonical text + provenance + license, for result assembly."""

    chunk_id: str
    text: str
    document_id: str
    source_kind: str
    standard_id: str | None
    locator: str | None
    license_class: str
    methods: list[tuple[str, str]] = field(default_factory=list)  # (method_id, method_version)


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

    @classmethod
    def create(
        cls, index: IndexSettings, embedding_dimension: int, embedder: Any = None
    ) -> SqliteIndexStore:
        """Open the §8 index from its config slice (connects + ensures schema). Pass an
        ``embedder`` (the ingestion/producer side) to record ``index_meta``; omit it for
        read-only use (retrieval)."""
        store = cls(
            index.db_path,
            embedding_dim=embedding_dimension,
            default_license_class=index.default_license_class,
        ).connect()
        if embedder is not None:
            store.write_index_meta(embedder)
        return store

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteIndexStore.connect() must be called first")
        return self._conn

    def connect(self) -> SqliteIndexStore:
        # SQLite won't create the parent directory — ensure it exists (no-op for ':memory:').
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
              license_class TEXT NOT NULL,
              content_hash  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
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
            "(document_id, title, source_kind, standard_id, doc_version, license_class, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                md.get("title"),
                md.get("source_kind") or "document",
                md.get("standard_id"),
                md.get("doc_version"),
                md.get("license_class") or self.default_license_class,
                md.get("content_hash"),
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

    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Document ids whose stored document-level ``content_hash`` matches — content dedup."""
        rows = self.conn.execute(
            "SELECT document_id FROM documents WHERE content_hash=?", (content_hash,)
        ).fetchall()
        return [r[0] for r in rows]

    async def delete_document(self, document_id: str) -> bool:
        """Remove the document and its chunks + vec/fts/method rows (virtual tables don't
        cascade). Returns True if the document existed."""
        old = [r[0] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE document_id=?", (document_id,)
        )]
        if old:
            marks = ",".join("?" * len(old))
            self.conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute(f"DELETE FROM method_chunks WHERE chunk_id IN ({marks})", old)
            self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        cur = self.conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
        self.conn.commit()
        return cur.rowcount > 0

    async def list_documents(self) -> list[dict[str, Any]]:
        """Inventory of indexed documents with chunk/embedding counts."""
        rows = self.conn.execute(
            "SELECT d.document_id, d.content_hash, "
            "(SELECT count(*) FROM chunks c WHERE c.document_id = d.document_id) AS chunk_count, "
            "(SELECT count(*) FROM vec_chunks v WHERE v.chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE document_id = d.document_id)) AS embedding_count "
            "FROM documents d"
        ).fetchall()
        return [
            {"document_id": r[0], "content_hash": r[1], "chunk_count": r[2], "embedding_count": r[3]}
            for r in rows
        ]

    # ---------------- read side (retrieval) ----------------

    def dense_knn(
        self, query_vec: list[float], k: int, filter: str | None = None
    ) -> list[Candidate]:
        """Exact KNN over ``vec_chunks`` (sqlite-vec), nearest first. ``filter`` is a permitted-
        chunk SQL predicate (Step B / license policy); unused in the dense-only slice."""
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(query_vec), k),
        ).fetchall()
        return [
            Candidate(chunk_id=cid, rank=i + 1, raw_score=dist)
            for i, (cid, dist) in enumerate(rows)
        ]

    def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """Fetch canonical text + provenance for the given chunk ids, preserving input order."""
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            "SELECT c.chunk_id, c.text, c.document_id, d.source_kind, d.standard_id, "
            "c.locator, c.license_class "
            "FROM chunks c JOIN documents d ON c.document_id = d.document_id "
            f"WHERE c.chunk_id IN ({marks})",
            chunk_ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        records: list[ChunkRecord] = []
        for cid in chunk_ids:
            r = by_id.get(cid)
            if r is None:
                continue
            methods = self.conn.execute(
                "SELECT method_id, method_version FROM method_chunks WHERE chunk_id=?", (cid,)
            ).fetchall()
            records.append(
                ChunkRecord(
                    chunk_id=r[0], text=r[1], document_id=r[2], source_kind=r[3],
                    standard_id=r[4], locator=r[5], license_class=r[6],
                    methods=[(m, v) for m, v in methods],
                )
            )
        return records
