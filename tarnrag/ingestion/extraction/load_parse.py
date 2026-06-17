"""LoadAndParseStage — the structured-extraction stage: a source becomes a ``StructuredDocument``.

It is a **container component**: its extractors are *children*, built through the one framework factory
in ``_build_children`` (exactly the way ``Pipeline`` builds its stage children), so the whole tree —
Pipeline → LoadAndParse → extractors — is assembled in one recursive pass from ``Settings.components``,
with no per-stage instantiation. ``Config.routes`` maps a ``source_kind`` to an extractor component
spec (``class_name`` + options), so choosing e.g. Docling for PDFs — with its options — is a config
edit; an unrouted kind falls back to ``default_extractor``. A per-document ``metadata['extractor']``
override is the one runtime choice a construction-time build can't pre-make, so it is built on demand
(once, shared with the routes).

The stage sets ``item.content = document.text`` (so the existing text path keeps working) and
``item.document`` (which the enrichers and the structure-aware chunker consume next).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import PipelineItem
from tarnrag.core.components import ComponentFactory
from tarnrag.ingestion.extraction.extractor import Extractor, Source
from tarnrag.ingestion.pipeline import PipelineStage

# Default source_kind -> extractor spec. Config-driven: override per format via Settings.components.
_DEFAULT_ROUTES: dict[str, dict[str, Any]] = {
    "text": {"class_name": "plain_text"},
    "txt": {"class_name": "plain_text"},
    "markdown": {"class_name": "markdown"},
    "md": {"class_name": "markdown"},
    "html": {"class_name": "html"},
    "htm": {"class_name": "html"},
    "pdf": {"class_name": "pdf_text"},  # the fast tier; route to {"class_name": "docling"} for high-fidelity
}


class LoadAndParseStage(PipelineStage):
    """Load + structured-parse a document, setting ``item.document`` + ``item.content = document.text``."""

    class Config(PipelineStage.Config):
        class_name: Literal["LoadAndParse"] = "LoadAndParse"
        # source_kind -> extractor spec (class_name + options); the config-driven route map.
        routes: dict[str, dict[str, Any]] = Field(
            default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_ROUTES.items()}
        )
        # fallback when the source_kind isn't routed.
        default_extractor: dict[str, Any] = Field(default_factory=lambda: {"class_name": "plain_text"})

    config: LoadAndParseStage.Config

    def __init__(self, config: LoadAndParseStage.Config) -> None:
        super().__init__(config)
        self._built: dict[str, Extractor] = {}  # canonical spec -> instance (one per distinct spec)
        self._by_kind: dict[str, Extractor] = {}  # source_kind -> its (shared) extractor child
        self._default: Extractor | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the route extractors as children, through the framework factory (the same hook
        ``Pipeline`` uses for its stages) — so they join the one recursive tree build."""
        self._built = {}
        self._by_kind = {kind: self._extractor(spec, factory) for kind, spec in self.config.routes.items()}
        self._default = self._extractor(self.config.default_extractor, factory)

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        self._ensure_children()
        md = item.metadata
        path = md.get("source_path")
        source_kind = self._infer_kind(path) or md.get("source_type") or ""
        document = self._select(source_kind, md.get("extractor")).extract(
            Source(
                source_id=md.get("source_id") or md.get("doc_id") or str(uuid.uuid4()),
                source_kind=source_kind,
                path=path,
                content=None if path else item.content,
                metadata=md,
            )
        )
        yield PipelineItem(
            content=document.text,
            metadata={**md, "doc_id": md.get("doc_id") or document.source_id, "loaded": True},
            document=document,
            provenance=item.provenance,
        )

    def _select(self, source_kind: str, override: str | None) -> Extractor:
        """The configured route's extractor (a prebuilt child), or — for the per-document override —
        the named extractor, built once on demand and shared with the routes."""
        if override:
            return self._extractor({"class_name": override}, ComponentFactory.get())
        return self._by_kind.get(source_kind, self._default)

    def _extractor(self, spec: dict[str, Any], factory: ComponentFactory) -> Extractor:
        """Build (or reuse) the extractor for ``spec`` via the factory; dedups identical specs so
        several routes share one instance."""
        key = json.dumps(spec, sort_keys=True)
        if key not in self._built:
            self._built[key] = factory.create_as(dict(spec), Extractor)
        return self._built[key]

    @staticmethod
    def _infer_kind(path: str | None) -> str:
        """The routing key from a file path: its extension (``txt``/``md``/``pdf``/``html``/…)."""
        return Path(path).suffix.lower().lstrip(".") if path else ""
