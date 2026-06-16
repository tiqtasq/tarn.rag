"""LoadAndParseStage — the structured-extraction stage: a source becomes a ``StructuredDocument``.

Routing is **config-driven**: ``Config.routes`` maps a ``source_kind`` to an extractor component spec
(``class_name`` + options), so choosing e.g. Docling for PDFs — with its options — is a config edit
(no code/call change), part of the pipeline in ``Settings.components``. A per-document
``metadata['extractor']`` override (a class_name) wins when present; an unrouted kind falls back to
``default_extractor`` (graceful degradation, FR-1.3). Extractor instances are **long-lived** —
instantiated once and cached on the stage (so e.g. Docling loads its models once, not per document).

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
from tarnrag.ingestion.extraction import Extractor, Source
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
        self._cache: dict[str, Extractor] = {}  # long-lived extractor instances, keyed by canonical spec

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        md = item.metadata
        path = md.get("source_path")
        source_kind = self._infer_kind(path) or md.get("source_type") or ""
        spec = (
            {"class_name": md["extractor"]} if md.get("extractor")
            else self.config.routes.get(source_kind, self.config.default_extractor)
        )
        document = self._extractor(spec).extract(
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

    def _extractor(self, spec: dict[str, Any]) -> Extractor:
        key = json.dumps(spec, sort_keys=True)
        if key not in self._cache:
            built = ComponentFactory.get().create(dict(spec))
            if not isinstance(built, Extractor):
                raise TypeError(
                    f"{spec.get('class_name')!r} is a {type(built).__name__}, not an Extractor"
                )
            self._cache[key] = built
        return self._cache[key]

    @staticmethod
    def _infer_kind(path: str | None) -> str:
        """The routing key from a file path: its extension (``txt``/``md``/``pdf``/``html``/…)."""
        return Path(path).suffix.lower().lstrip(".") if path else ""
