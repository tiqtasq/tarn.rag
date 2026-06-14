"""LoadAndParseStage — load a document from a file (or use given content).

The PDF backend is pluggable: the stage holds a registry of parser strategies (built once —
stage config), and a request picks one per call via ``metadata['parser']`` (item data, flows
inline). Unknown/absent → the default. txt/html have a single obvious loader and ignore
``parser``. Stages stay pure (D6): this just dispatches on its input.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from tarnrag.ingestion.pipeline import MapperStage
from tarnrag.ingestion.stages.parsers import (
    DEFAULT_PDF_PARSER,
    DEFAULT_PDF_PARSERS,
    load_html,
)


class LoadAndParseStage(MapperStage):
    """
    If ``metadata['source_path']`` is set, load and parse the file; otherwise treat
    the incoming content as already loaded. Assigns a provisional ``doc_id`` (the
    DocumentResultSink overwrites it with the stored id).
    """

    def __init__(
        self,
        supported_types: list[str] | None = None,
        pdf_parsers: dict[str, Callable[[str], str]] | None = None,
        default_pdf_parser: str = DEFAULT_PDF_PARSER,
    ):
        # Set before super().__init__(), which runs validate().
        types = supported_types or ["txt", "pdf", "html"]
        self.supported_types = types
        self.pdf_parsers = pdf_parsers or dict(DEFAULT_PDF_PARSERS)
        self.default_pdf_parser = default_pdf_parser
        super().__init__(name="LoadAndParse")

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        source_path = metadata.get("source_path")
        content = (
            self._load_file(source_path, metadata.get("parser")) if source_path else text
        )
        return content, {"doc_id": metadata.get("doc_id") or str(uuid.uuid4()), "loaded": True}

    def _load_file(self, path: str, parser: str | None = None) -> str:
        """
        Read a file by extension: txt/md plain, pdf via the selected parser, html via ``load_html``.
        """
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("txt", "text", "md"):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        if ext == "pdf":
            return self._pdf_loader(parser)(path)
        if ext in ("html", "htm"):
            return load_html(path)
        raise ValueError(f"Unsupported file type: {ext!r}")

    def _pdf_loader(self, parser: str | None) -> Callable[[str], str]:
        name = parser or self.default_pdf_parser
        try:
            return self.pdf_parsers[name]
        except KeyError:
            raise ValueError(
                f"Unknown pdf parser {name!r}; available: {sorted(self.pdf_parsers)}"
            ) from None

    def validate(self) -> None:
        if self.default_pdf_parser not in self.pdf_parsers:
            raise ValueError(
                f"default_pdf_parser {self.default_pdf_parser!r} not in registry "
                f"{sorted(self.pdf_parsers)}"
            )
