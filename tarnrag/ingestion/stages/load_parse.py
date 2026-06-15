"""LoadAndParseStage — load a document from a file (or use given content).

The PDF backend is pluggable: the loaders live in ``stages/parsers.py`` (a name -> callable
registry), and a request picks one per call via ``metadata['parser']`` (item data, flows inline).
Unknown/absent → the config's ``default_pdf_parser``. txt/html have a single obvious loader and
ignore ``parser``. Stages stay pure (D6): this just dispatches on its input.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Literal

from pydantic import model_validator

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

    class Config(MapperStage.Config):
        class_name: Literal["LoadAndParse"] = "LoadAndParse"
        supported_types: list[str] = ["txt", "pdf", "html"]
        default_pdf_parser: str = DEFAULT_PDF_PARSER

        @model_validator(mode="after")
        def _known_pdf_parser(self) -> LoadAndParseStage.Config:
            if self.default_pdf_parser not in DEFAULT_PDF_PARSERS:
                raise ValueError(
                    f"default_pdf_parser {self.default_pdf_parser!r} not in registry "
                    f"{sorted(DEFAULT_PDF_PARSERS)}"
                )
            return self

    config: LoadAndParseStage.Config

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
        name = parser or self.config.default_pdf_parser
        try:
            return DEFAULT_PDF_PARSERS[name]
        except KeyError:
            raise ValueError(
                f"Unknown pdf parser {name!r}; available: {sorted(DEFAULT_PDF_PARSERS)}"
            ) from None
