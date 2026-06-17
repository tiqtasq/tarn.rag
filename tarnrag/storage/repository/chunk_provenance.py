"""Chunk-provenance (de)serialization — the layout-aware columns that ride beside the §8 chunk row.

The geometry / header-path / auto-merging-tree provenance, plus the cell- and annotation-level child
tables, are nested + variable, so they persist as JSON (or normalized child rows) rather than §8 scalar
columns. This module is the one place that maps ``ChunkProvenance`` / ``Table`` / ``Annotation`` ↔ rows,
in both directions, so ``base.py`` keeps only the SQL orchestration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from tarnrag.contracts import Annotation, ChunkProvenance, Span, Table, TableCell

_GEOMETRY = TypeAdapter(list[Span])


def encode_geometry(geometry: list[Span]) -> str:
    """A ``Geometry`` (char spans + page boxes) as a JSON string for a ``geometry`` column."""
    return _GEOMETRY.dump_json(geometry).decode("utf-8")


def decode_geometry(blob: str) -> list[Span]:
    """Inverse of ``encode_geometry``."""
    return list(_GEOMETRY.validate_json(blob))


def provenance_columns(provenance: ChunkProvenance | None, parent_chunk_id: str | None) -> dict[str, Any]:
    """The chunk row's layout-aware provenance columns (header_path / level / geometry + the resolved
    ``parent_chunk_id``). Inverse: ``row_to_provenance``."""
    return {
        "header_path": json.dumps(provenance.header_path) if provenance else None,
        "level": provenance.level if provenance else 0,
        "parent_chunk_id": parent_chunk_id,
        "geometry": encode_geometry(provenance.geometry) if provenance and provenance.geometry else None,
    }


def row_to_provenance(row: Mapping[str, Any]) -> ChunkProvenance:
    """Rebuild the persisted provenance from a chunk row. The ``table`` is attached separately from
    ``table_cells``; ``source_element_ids`` / ``annotations`` are not persisted under §8."""
    return ChunkProvenance(
        header_path=json.loads(row["header_path"]) if row["header_path"] else [],
        geometry=decode_geometry(row["geometry"]) if row["geometry"] else [],
        content_hash=row["content_hash"],
        level=row["level"] or 0,
        parent_chunk_id=row["parent_chunk_id"],
    )


def table_cell_rows(chunk_id: str, table: Table) -> list[dict[str, Any]]:
    """The ``table_cells`` rows for a table chunk's cells."""
    return [
        {
            "chunk_id": chunk_id,
            "cell_id": cell.id,
            "row": cell.row,
            "col": cell.col,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "is_column_header": int(cell.is_column_header),
            "is_row_header": int(cell.is_row_header),
            "text": cell.text,
            "geometry": encode_geometry(cell.geometry) if cell.geometry else None,
        }
        for cell in table.cells
    ]


def rebuild_table(rows: list[Mapping[str, Any]]) -> Table:
    """Rebuild a ``Table`` from its ``table_cells`` rows; ``n_rows`` / ``n_cols`` derive from the cells,
    and the rendered markdown is the chunk's ``text`` (so it stays empty here)."""
    cells = [
        TableCell(
            id=r["cell_id"],
            row=r["row"],
            col=r["col"],
            row_span=r["row_span"],
            col_span=r["col_span"],
            is_column_header=bool(r["is_column_header"]),
            is_row_header=bool(r["is_row_header"]),
            text=r["text"],
            geometry=decode_geometry(r["geometry"]) if r["geometry"] else [],
        )
        for r in rows
    ]
    n_rows = max((c.row + c.row_span for c in cells), default=0)
    n_cols = max((c.col + c.col_span for c in cells), default=0)
    return Table(n_rows=n_rows, n_cols=n_cols, cells=cells)


def annotation_rows(chunk_id: str, annotations: list[Annotation]) -> list[dict[str, Any]]:
    """The ``chunk_annotations`` rows for a chunk's annotations (position-ordered)."""
    return [
        {
            "chunk_id": chunk_id,
            "ordinal": i,
            "producer": a.producer,
            "type": a.type,
            "value": json.dumps(a.value),
            "span": encode_geometry(a.span) if a.span else None,
            "deterministic": int(a.deterministic),
        }
        for i, a in enumerate(annotations)
    ]


def row_to_annotation(row: Mapping[str, Any]) -> Annotation:
    """Rebuild an ``Annotation`` from a ``chunk_annotations`` row."""
    return Annotation(
        producer=row["producer"],
        type=row["type"],
        value=json.loads(row["value"]),
        span=decode_geometry(row["span"]) if row["span"] else None,
        deterministic=bool(row["deterministic"]),
    )
