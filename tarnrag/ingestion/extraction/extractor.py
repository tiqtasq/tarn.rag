"""Extractor seam — a ``Source`` becomes a ``StructuredDocument``.

Extractors are config-driven ``Component``s (registered by ``class_name``, like the pipeline stages),
so the document-processing pipeline (extract → enrich) is configurable as data and a user can drop in
their own extractor. ``extract`` routes a ``Source`` to the right extractor by ``source_kind`` — an
explicit ``metadata['extractor']`` override wins, else the per-kind default, else plain text (graceful
degradation). Tiered fast/high-fidelity routing slots in here as the route values grow.

See ``doc/extraction-seam-design.md``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tarnrag.contracts import StructuredDocument
from tarnrag.core.components import Component, ComponentFactory


class Source(BaseModel):
    """An extractor's input: where the bytes/text are, the document identity, and routing hints."""

    source_id: str
    source_kind: str = ""  # "pdf" | "markdown" | "text" | … (detected/declared by the caller)
    path: str | None = None  # filesystem path, if loading from disk
    content: str | None = None  # already-loaded text, if not from disk
    metadata: dict[str, Any] = Field(default_factory=dict)  # hints, e.g. {"extractor": "plain_text"}


class Extractor(Component):
    """Port: produce a ``StructuredDocument`` from a ``Source``. Concrete extractors pin ``class_name``."""

    class Config(Component.Config):
        """Base extractor config; concrete extractors pin ``class_name`` and add their own options."""

    @abstractmethod
    def extract(self, source: Source) -> StructuredDocument:
        """Parse the source into a structured document."""

    @staticmethod
    def _read_text(source: Source) -> str:
        """The source's text: inline ``content`` if given, else read ``path`` (utf-8), else empty."""
        if source.content is not None:
            return source.content
        if source.path:
            return Path(source.path).read_text(encoding="utf-8", errors="replace")
        return ""


# format -> extractor class_name. Tiered routing (fast/high-fidelity) extends this to per-format maps.
DEFAULT_ROUTES: dict[str, str] = {
    "text": "plain_text",
    "txt": "plain_text",
    "markdown": "markdown",
    "md": "markdown",
    "html": "html",
    "htm": "html",
    "pdf": "pdf_text",  # fast born-digital tier; high-fidelity Docling is opt-in (a follow-on)
}


def extract(source: Source) -> StructuredDocument:
    """Route a ``Source`` to its extractor and run it: an explicit ``metadata['extractor']`` wins, else
    ``DEFAULT_ROUTES[source_kind]``, else the plain-text fallback (graceful degradation, FR-1.3)."""
    name = source.metadata.get("extractor") or DEFAULT_ROUTES.get(source.source_kind, "plain_text")
    extractor = ComponentFactory.get().create({"class_name": name})
    if not isinstance(extractor, Extractor):
        raise TypeError(f"{name!r} is a {type(extractor).__name__}, not an Extractor")
    return extractor.extract(source)
