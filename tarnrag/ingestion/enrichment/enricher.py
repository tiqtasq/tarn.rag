"""The Enricher component family: annotate a ``StructuredDocument`` after extraction (FR-5).

An ``Enricher`` is a pure ``Component`` that appends typed ``Annotation``s to a document's elements (or
the document itself) — entities, topics, classifications, … — each recording its ``producer`` and a
deterministic-vs-generative flag (anti-hallucination). The ``EnrichStage`` runs a configured, ordered
list of them over ``item.document``; the annotations then flow into chunks (the chunker aggregates
element annotations into ``ChunkProvenance``). Users register their own (NER/topic) behind this port on
the same framework — no base-class edits (AC-2).
"""

from __future__ import annotations

from abc import abstractmethod

from tarnrag.contracts import Element, Span, StructuredDocument
from tarnrag.core.components import Component


class Enricher(Component):
    """Port: annotate a ``StructuredDocument`` in place (append to ``element.annotations`` and/or
    ``document.annotations``). Runs after extraction, before chunking."""

    class Config(Component.Config):
        """Base enricher config; concrete enrichers pin ``class_name`` and add their own options."""

    @abstractmethod
    def enrich(self, document: StructuredDocument) -> None:
        """Append annotations to the document and/or its elements (mutates in place)."""

    @staticmethod
    def _subspan(element: Element, start: int, end: int) -> Span:
        """The absolute document span of ``element.text[start:end]`` — the element's char offset plus
        the sub-range — so a span-level annotation indexes ``StructuredDocument.text`` like any Span."""
        base = element.geometry[0].start if element.geometry else 0
        return Span(start=base + start, end=base + end)
