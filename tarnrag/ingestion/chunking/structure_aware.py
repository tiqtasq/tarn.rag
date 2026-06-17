"""Structure-aware chunker — split on the document's structure, build the auto-merging tree.

Walks the ``StructuredDocument`` elements in reading order and groups them into **sections** by header
path (a heading + the body elements beneath it). Within a section: a **table is its own atomic leaf**
(never split, never merged), and consecutive text/code elements **pack into leaf chunks** under a
``max_chars`` budget without crossing the section boundary. When a section yields more than one leaf it
also emits a **section parent** (``level`` 1) over them — the auto-merging tree: retrieve a leaf, merge
up to its parent for context. Each chunk carries the section ``header_path``, the source element ids,
the aggregated geometry (page boxes included), and — for a table chunk — the ``Table`` itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tarnrag.contracts import ChunkProvenance, Element, ElementKind, Span, StructuredDocument
from tarnrag.ingestion.chunking.chunker import Chunker

_JOIN = "\n\n"  # joins packed element texts within a chunk


class StructureAwareChunker(Chunker):
    """Split on document structure (sections / tables), emitting a leaf + section-parent tree."""

    class Config(Chunker.Config):
        class_name: Literal["structure_aware"] = "structure_aware"
        max_chars: int = Field(default=1200, gt=0)  # soft budget for packing text elements into a leaf

    config: StructureAwareChunker.Config

    def chunk(self, document: StructuredDocument) -> list[Chunker.TempChunk]:
        chunks: list[Chunker.TempChunk] = []
        for members in self._sections(document.elements):
            self._emit_section(members, chunks)
        if not chunks and document.text.strip():  # no usable structure → one leaf over the whole text
            chunks.append(
                Chunker.TempChunk(
                    text=document.text,
                    provenance=ChunkProvenance(
                        source_element_ids=[e.id for e in document.elements],
                        geometry=[Span(start=0, end=len(document.text))],
                        content_hash=self._hash(document.text),
                    ),
                )
            )
        return chunks

    @staticmethod
    def _sections(elements: list[Element]) -> list[list[Element]]:
        """Group elements into sections: a heading starts a new one, and its body elements (whose
        ``header_path`` equals the heading's full path) fall in with it until the next heading."""
        sections: list[list[Element]] = []
        current: list[Element] = []
        current_key: tuple[str, ...] | None = None
        for el in elements:
            key = tuple(el.header_path + [el.text]) if el.kind == ElementKind.HEADING else tuple(el.header_path)
            if el.kind == ElementKind.HEADING or (current and key != current_key):
                if current:
                    sections.append(current)
                current, current_key = [el], key
            else:
                if not current:
                    current_key = key
                current.append(el)
        if current:
            sections.append(current)
        return sections

    def _emit_section(self, members: list[Element], chunks: list[Chunker.TempChunk]) -> None:
        """Emit a section's chunks: its packed leaves, plus a section parent over them when >1 leaf."""
        heading = members[0] if members[0].kind == ElementKind.HEADING else None
        section_path = (heading.header_path + [heading.text]) if heading else members[0].header_path
        body = [m for m in members if m.kind != ElementKind.HEADING]
        leaves = self._leaves(body, section_path)
        if not leaves:  # heading-only section — captured in its descendants' header_path
            return
        if len(leaves) == 1:  # nothing to merge up to — keep it flat
            chunks.append(leaves[0])
            return
        parent_index = len(chunks)
        heading_text = [heading.text] if heading else []
        parent_text = _JOIN.join(heading_text + [leaf.text for leaf in leaves])
        chunks.append(self._parent(parent_text, members, section_path))
        for leaf in leaves:
            leaf.parent_index = parent_index
            chunks.append(leaf)

    def _leaves(self, body: list[Element], section_path: list[str]) -> list[Chunker.TempChunk]:
        """Pack a section's body into leaves: tables stay atomic; text/code packs under ``max_chars``."""
        leaves: list[Chunker.TempChunk] = []
        buf_text: list[str] = []
        buf_els: list[Element] = []
        size = 0

        def flush() -> None:
            nonlocal buf_text, buf_els, size
            if buf_els:
                leaves.append(self._leaf(_JOIN.join(buf_text), buf_els, section_path))
                buf_text, buf_els, size = [], [], 0

        for el in body:
            if el.kind == ElementKind.TABLE:
                flush()
                leaves.append(self._table_leaf(el, section_path))
                continue
            if not el.text:
                continue
            if len(el.text) > self.config.max_chars:  # oversize single element → character-split it
                flush()
                leaves.extend(self._oversize(el, section_path))
                continue
            if buf_els and size + len(el.text) > self.config.max_chars:
                flush()
            buf_text.append(el.text)
            buf_els.append(el)
            size += len(el.text)
        flush()
        return leaves

    def _leaf(self, text: str, elements: list[Element], header_path: list[str]) -> Chunker.TempChunk:
        """A level-0 leaf over ``elements``, packed into ``text``."""
        return Chunker.TempChunk(
            text=text,
            provenance=ChunkProvenance(
                source_element_ids=[e.id for e in elements],
                geometry=self._union_geometry(elements),
                header_path=header_path,
                content_hash=self._hash(text),
                annotations=self._annotations(elements),
            ),
        )

    def _table_leaf(self, element: Element, header_path: list[str]) -> Chunker.TempChunk:
        """A table's atomic leaf — its markdown is the searchable text; the ``Table`` rides on the
        provenance so cell geometry + header addressing can be persisted (the table_cells slice)."""
        text = element.text or (element.table.markdown if element.table else "")
        return Chunker.TempChunk(
            text=text,
            provenance=ChunkProvenance(
                source_element_ids=[element.id],
                geometry=self._union_geometry([element]),
                header_path=header_path,
                content_hash=self._hash(text),
                annotations=self._annotations([element]),
                table=element.table,
            ),
        )

    def _oversize(self, element: Element, header_path: list[str]) -> list[Chunker.TempChunk]:
        """Character-split a single element that alone exceeds ``max_chars`` (e.g. a huge paragraph).
        The pieces share the element's geometry — box-precise sub-spans are a later refinement."""
        size = self.config.max_chars
        text = element.text
        return [
            Chunker.TempChunk(
                text=text[i : i + size],
                provenance=ChunkProvenance(
                    source_element_ids=[element.id],
                    geometry=self._union_geometry([element]),
                    header_path=header_path,
                    content_hash=self._hash(text[i : i + size]),
                    annotations=self._annotations([element]),
                ),
            )
            for i in range(0, len(text), size)
        ]

    def _parent(self, text: str, members: list[Element], header_path: list[str]) -> Chunker.TempChunk:
        """The section parent (``level`` 1) over a whole section — the auto-merge target for its leaves."""
        return Chunker.TempChunk(
            text=text,
            provenance=ChunkProvenance(
                source_element_ids=[m.id for m in members],
                geometry=self._union_geometry(members),
                header_path=header_path,
                content_hash=self._hash(text),
                level=1,
                annotations=self._annotations(members),
            ),
        )
