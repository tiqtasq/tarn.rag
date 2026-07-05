"""Shared passage rendering — the ONE place a retrieved hit becomes reader-facing text (P1 PR-2).

Every component that lays evidence before the LLM (the reasoners' numbered passages, the grounding
checkers' cited passages) goes through :func:`passage_text`, so the reader and its fact-checker
always see the same representation. For a chunk carrying a real ``Table``, the ``structured`` view
renders ``Table.contextual_text()`` — each value bound to its row/column headers — instead of the
raw grid text: the measured table penalty (attribution 0.88 vs text 0.99) is the reader mis-reading
grid alignment, so meaning is spelled out. ``text`` is the raw stored text (the pre-P1 behavior;
evals pin one view explicitly — never inherit the default).
"""

from __future__ import annotations

from tarnrag.contracts import RetrievalResult

# The reader-facing views: 'structured' renders tables from their cells; 'text' is the stored text.
TABLE_VIEWS = ("structured", "text")


def passage_text(result: RetrievalResult, table_view: str = "structured") -> str:
    """The reader-facing text for one hit: under ``structured``, a table chunk renders as its
    header-contextualized lines (falling back to the stored text when the table has no data cells);
    everything else — and everything under ``text`` — is the stored text unchanged."""
    if table_view == "structured" and result.provenance and result.provenance.table:
        return result.provenance.table.contextual_text() or result.text
    return result.text
