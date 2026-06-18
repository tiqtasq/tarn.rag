"""PostgreSQL adapter (SQLAlchemy Core + asyncpg + pgvector).

§8 dense retrieval (``dense_knn``) via the pgvector ``<=>`` cosine-distance operator. Requires the
``postgres`` extra (``asyncpg``, ``pgvector``); not imported on the SQLite path.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection
from pgvector.sqlalchemy import Vector

from tarnrag.storage.repository.base import DocumentRepository
from tarnrag.contracts import Candidate, ChunkRecord


class PostgresRepository(DocumentRepository):
    """
    PostgreSQL adapter. Inherits the portable upsert + CRUD from the base; supplies
    the asyncpg driver URL, the pgvector column, and pgvector cosine search.
    """

    def _driver_url(self, url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    def _vector_type(self):
        return Vector(self.embedding_dimension)

    def _encode_vector(self, vector: list[float]):
        return list(vector)  # pgvector accepts a Python list directly

    def _decode_vector(self, stored) -> list[float]:
        return list(stored)

    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        # document_id is the documents PK (already unique). The pgvector ANN index + the sparse FTS
        # GIN index (matching sparse_search's to_tsvector expression so the planner uses it).
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector "
            "ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)"
        )
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (to_tsvector('english', text))"
        )

    async def dense_knn(self, query_vec: list[float], k: int) -> list[Candidate]:
        """
        §8 dense KNN over the pgvector ``embeddings`` table (cosine distance, nearest first),
        returned as ranked ``Candidate``s — the SQLite ``vec_chunks`` counterpart.
        """
        dist = self.embeddings.c.vector.cosine_distance(query_vec)
        stmt = select(self.embeddings.c.chunk_id, dist.label("distance")).order_by(dist).limit(k)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [Candidate(chunk_id=cid, rank=i + 1, raw_score=d) for i, (cid, d) in enumerate(rows)]

    async def sparse_search(self, query_text: str, k: int) -> list[Candidate]:
        """§8 sparse retrieval over a ``to_tsvector('english', text)`` GIN index (``ts_rank_cd``, best
        first). ``plainto_tsquery`` parses the raw text safely; ``raw_score`` is ts_rank (higher better)."""
        tsv = func.to_tsvector("english", self.chunks.c.text)
        tsq = func.plainto_tsquery("english", query_text)
        score = func.ts_rank_cd(tsv, tsq).label("score")
        stmt = select(self.chunks.c.chunk_id, score).where(tsv.op("@@")(tsq)).order_by(score.desc()).limit(k)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [Candidate(chunk_id=cid, rank=i + 1, raw_score=s) for i, (cid, s) in enumerate(rows)]

    async def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """
        §8 hydration via SQLAlchemy over the normal tables (chunks joined to documents, plus
        method_chunks), preserving input order. No virtual tables here, so unlike SQLite this
        stays in Core.
        """
        if not chunk_ids:
            return []
        c, d, m = self.chunks, self.documents, self.method_chunks
        stmt = (
            select(
                c.c.chunk_id, c.c.text, c.c.document_id, d.c.source_kind,
                d.c.standard_id, c.c.locator, c.c.license_class,
                c.c.ai_grounding_allowed, c.c.available,
            )
            .select_from(c.join(d, c.c.document_id == d.c.document_id))
            .where(c.c.chunk_id.in_(chunk_ids))
        )
        async with self.engine.connect() as conn:
            by_id = {r[0]: r for r in (await conn.execute(stmt)).all()}
            prov = await self._chunk_provenance(conn, chunk_ids)
            records: list[ChunkRecord] = []
            for cid in chunk_ids:
                r = by_id.get(cid)
                if r is None:
                    continue
                methods = (
                    await conn.execute(
                        select(m.c.method_id, m.c.method_version).where(m.c.chunk_id == cid)
                    )
                ).all()
                records.append(self._to_chunk_record(r, methods, prov.get(cid)))
        return records
