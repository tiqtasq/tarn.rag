"""HTML extractor — structure-aware: a ``Source`` → ``StructuredDocument``.

Walks the parsed DOM into reading-order blocks — headings (``<h1..6>`` → levels), paragraphs, list items
(``<li>``), code (``<pre>``), and tables (``<table>`` → ``Table``/``TableCell``, ``<th>`` → header cells)
— then assembles them like the markdown extractor (shared ``assemble``). ``beautifulsoup4`` is the
optional ``parsers`` extra, imported lazily.
"""

from __future__ import annotations

from typing import Literal

from tarnrag.contracts import StructuredDocument, Table, TableCell
from tarnrag.ingestion.components.extraction._blocks import Block, assemble
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP = {"script", "style", "head"}  # non-content subtrees


class HtmlExtractor(Extractor):
    """Structure-aware HTML extraction (headings / paragraphs / lists / code / tables)."""

    class Config(Extractor.Config):
        class_name: Literal["html"] = "html"

    config: HtmlExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self._read_text(source), "html.parser")
        blocks: list[Block] = []
        self._walk(soup.body or soup, blocks)  # a full doc has <body>; a fragment is walked whole
        elements, text = assemble(blocks)
        return self._create_document(source, text, elements, extractor=self.config.class_name, default_kind="html")

    @staticmethod
    def _walk(node: object, blocks: list[Block]) -> None:
        """Depth-first DOM walk: emit a block for each known structural tag (and don't recurse into it);
        recurse into containers (``<div>``/``<section>``/…); capture loose text as paragraphs."""
        from bs4 import NavigableString

        for child in node.children:  # type: ignore[attr-defined]
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    blocks.append(("paragraph", text, None, None))
                continue
            name = child.name
            if name in _SKIP:
                continue
            if name in _HEADINGS:
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(("heading", text, int(name[1]), None))
            elif name == "p":
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(("paragraph", text, None, None))
            elif name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        blocks.append(("list_item", text, None, None))
            elif name == "table":
                table, markdown = HtmlExtractor._table(child)
                if table.cells:
                    blocks.append(("table", markdown, None, table))
            elif name == "pre":
                text = child.get_text().strip("\n")
                if text.strip():
                    blocks.append(("code", text, None, None))
            else:
                HtmlExtractor._walk(child, blocks)

    @staticmethod
    def _table(table_tag: object) -> tuple[Table, str]:
        """Parse a ``<table>`` into a ``Table`` (``<th>`` in row 0 → column header, in col 0 → row header)
        + a markdown rendering (the table chunk's searchable text)."""
        grid: list[list[str]] = []
        cells: list[TableCell] = []
        for r, tr in enumerate(table_tag.find_all("tr")):  # type: ignore[attr-defined]
            row_texts: list[str] = []
            for c, cell in enumerate(tr.find_all(["td", "th"], recursive=False)):
                text = cell.get_text(" ", strip=True)
                is_th = cell.name == "th"
                cells.append(
                    TableCell(
                        id=f"r{r}c{c}", row=r, col=c, text=text,
                        is_column_header=is_th and r == 0,
                        is_row_header=is_th and r > 0 and c == 0,
                        row_span=int(cell.get("rowspan", 1) or 1),
                        col_span=int(cell.get("colspan", 1) or 1),
                    )
                )
                row_texts.append(text)
            grid.append(row_texts)
        markdown = HtmlExtractor._render(grid)
        n_cols = max((len(row) for row in grid), default=0)
        return Table(n_rows=len(grid), n_cols=n_cols, cells=cells, markdown=markdown), markdown

    @staticmethod
    def _render(grid: list[list[str]]) -> str:
        """Render a cell grid as a markdown table (header = row 0)."""
        if not grid:
            return ""
        n_cols = max(len(row) for row in grid)

        def line(cells: list[str]) -> str:
            return "| " + " | ".join(cells + [""] * (n_cols - len(cells))) + " |"

        out = [line(grid[0]), "| " + " | ".join(["---"] * n_cols) + " |"]
        out += [line(row) for row in grid[1:]]
        return "\n".join(out)
