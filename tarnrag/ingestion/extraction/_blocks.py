"""Shared substrate for the text extractors (markdown / HTML).

Both parse their source into a flat list of ``Block``s — one per heading / paragraph / code / list item
/ table, in reading order — then ``assemble`` turns those into ``Element``s over a normalized text:
threading the open-heading stack into each element's ``header_path`` / ``parent_id`` and giving it a
char-offset ``Span`` into that text. The load-bearing invariant: ``doc.text[span] == element.text`` for
every element.
"""

from __future__ import annotations

from tarnrag.contracts import Element, ElementKind, Span, Table

# (kind, normalized_text, heading_level | None, table | None). kind ∈ heading/paragraph/code/list_item/table.
Block = tuple[str, str, int | None, Table | None]

_KIND = {
    "code": ElementKind.CODE,
    "list_item": ElementKind.LIST_ITEM,
    "table": ElementKind.TABLE,
    "paragraph": ElementKind.PARAGRAPH,
}
_SEP = "\n\n"  # joins block texts into the normalized document text


def assemble(blocks: list[Block]) -> tuple[list[Element], str]:
    """Build elements + the normalized text from blocks, assigning each element a span into that text
    and the header path of its enclosing headings (``parent_id`` = the deepest enclosing heading)."""
    elements: list[Element] = []
    stack: list[tuple[int, str, str]] = []  # (level, text, element_id) of open headings
    parts: list[str] = []
    offset = 0
    for idx, (kind, text, level, table) in enumerate(blocks):
        start = offset
        end = start + len(text)
        offset = end + len(_SEP)
        parts.append(text)
        eid = f"e{idx}"
        if kind == "heading" and level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
            elements.append(
                Element(
                    id=eid, kind=ElementKind.HEADING, text=text, level=level,
                    header_path=[t for _, t, _ in stack],
                    parent_id=stack[-1][2] if stack else None,
                    geometry=[Span(start=start, end=end)],
                )
            )
            stack.append((level, text, eid))
        else:
            elements.append(
                Element(
                    id=eid, kind=_KIND[kind], text=text,
                    header_path=[t for _, t, _ in stack],
                    parent_id=stack[-1][2] if stack else None,
                    geometry=[Span(start=start, end=end)],
                    table=table,
                )
            )
    return elements, _SEP.join(parts)
