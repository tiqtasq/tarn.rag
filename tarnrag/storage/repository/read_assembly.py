"""Read-side assembly for the repository — fetch a chunk's child rows, build its read models.

The repository's write half maps DTOs to rows; this companion owns the READ half's assembly:
the batched child-table fetches (``table_cells`` / ``chunk_annotations`` / ``method_chunks``) and
the construction of ``ChunkProvenance`` / ``ChunkRecord`` from them (via the pure row↔model
converters in ``chunk_provenance``). Split out of ``base.py`` so ``DocumentRepository`` stays the
port implementation and this stays the one home of read-model assembly — both dialects' ``hydrate``
and the base's ``get_chunk`` / ``get_chunks_by_document`` go through it (``repo.reads``).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncConnection

from tarnrag.contracts import Chunk, ChunkProvenance, ChunkRecord
from tarnrag.storage.repository import chunk_provenance as cp


class ReadAssembler:
    """Assemble chunk read models over the §8 child tables (held by reference; defined in base)."""

    def __init__(
        self, chunks: Table, table_cells: Table, chunk_annotations: Table, method_chunks: Table
    ) -> None:
        self._chunks = chunks
        self._table_cells = table_cells
        self._chunk_annotations = chunk_annotations
        self._method_chunks = method_chunks

    async def attach_tables(self, conn: AsyncConnection, chunks: list[Chunk]) -> None:
        """Rebuild each table chunk's ``provenance.table`` from its ``table_cells`` rows (one query)."""
        grouped = await self.child_rows_by_chunk(
            conn,
            self._table_cells,
            [c.id for c in chunks if c.id],
            self._table_cells.c.row,
            self._table_cells.c.col,
        )
        for chunk in chunks:
            if (rows := grouped.get(chunk.id)) and chunk.provenance is not None:
                chunk.provenance.table = cp.rebuild_table(rows)

    async def attach_annotations(self, conn: AsyncConnection, chunks: list[Chunk]) -> None:
        """Rebuild each chunk's ``provenance.annotations`` from its ``chunk_annotations`` rows (one query)."""
        grouped = await self.child_rows_by_chunk(
            conn,
            self._chunk_annotations,
            [c.id for c in chunks if c.id],
            self._chunk_annotations.c.ordinal,
        )
        for chunk in chunks:
            if (rows := grouped.get(chunk.id)) and chunk.provenance is not None:
                chunk.provenance.annotations = [cp.row_to_annotation(r) for r in rows]

    async def chunk_provenance(
        self, conn: AsyncConnection, chunk_ids: list[str]
    ) -> dict[str, ChunkProvenance]:
        """``ChunkProvenance`` per chunk id — provenance columns + ``table_cells`` + ``chunk_annotations``.
        The shared, dialect-agnostic provenance fetch behind both dialects' ``hydrate``."""
        if not chunk_ids:
            return {}
        c = self._chunks.c
        rows = (
            await conn.execute(
                select(c.chunk_id, c.header_path, c.level, c.parent_chunk_id, c.geometry, c.content_hash)
                .where(c.chunk_id.in_(chunk_ids))
            )
        ).mappings().all()
        cells = await self.child_rows_by_chunk(
            conn, self._table_cells, chunk_ids, self._table_cells.c.row, self._table_cells.c.col
        )
        anns = await self.child_rows_by_chunk(
            conn, self._chunk_annotations, chunk_ids, self._chunk_annotations.c.ordinal
        )
        provenance: dict[str, ChunkProvenance] = {}
        for r in rows:
            prov = cp.row_to_provenance(r)
            if cell_rows := cells.get(r["chunk_id"]):
                prov.table = cp.rebuild_table(cell_rows)
            if ann_rows := anns.get(r["chunk_id"]):
                prov.annotations = [cp.row_to_annotation(a) for a in ann_rows]
            provenance[r["chunk_id"]] = prov
        return provenance

    async def methods_by_chunk(
        self, conn: AsyncConnection, chunk_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """``(method_id, method_version)`` pairs per chunk id, in ONE query (ordered for determinism) —
        the batched method fetch behind both dialects' ``hydrate`` (formerly a per-chunk N+1)."""
        if not chunk_ids:
            return {}
        mc = self._method_chunks.c
        rows = (
            await conn.execute(
                select(mc.chunk_id, mc.method_id, mc.method_version)
                .where(mc.chunk_id.in_(chunk_ids))
                .order_by(mc.method_id, mc.method_version)
            )
        ).all()
        grouped: dict[str, list[tuple[str, str]]] = {}
        for cid, mid, ver in rows:
            grouped.setdefault(cid, []).append((mid, ver))
        return grouped

    async def child_rows_by_chunk(
        self, conn: AsyncConnection, table: Table, chunk_ids: list[str], *order_by: Any
    ) -> dict[str, list[Any]]:
        """Fetch a chunk-child table's rows for ``chunk_ids``, grouped by ``chunk_id`` (ordered by
        ``order_by``) — the shared fetch behind the ``attach_*`` / provenance reads."""
        if not chunk_ids:
            return {}
        rows = (
            await conn.execute(select(table).where(table.c.chunk_id.in_(chunk_ids)).order_by(*order_by))
        ).mappings().all()
        grouped: dict[str, list[Any]] = {}
        for r in rows:
            grouped.setdefault(r["chunk_id"], []).append(r)
        return grouped

    @staticmethod
    def create_chunk_record(row, methods, provenance: ChunkProvenance | None) -> ChunkRecord:
        """Assemble a ``ChunkRecord`` from a hydrate row + its method refs + provenance — the one place
        the field list lives. Both dialects' ``hydrate`` queries select the same column order:
        (chunk_id, text, document_id, source_kind, standard_id, locator, license_class,
        ai_grounding_allowed, available), so the positional row is shared."""
        return ChunkRecord(
            chunk_id=row[0], text=row[1], document_id=row[2], source_kind=row[3],
            standard_id=row[4], locator=row[5], license_class=row[6],
            ai_grounding_allowed=bool(row[7]), available=bool(row[8]),
            methods=[(mid, ver) for mid, ver in methods], provenance=provenance,
        )
