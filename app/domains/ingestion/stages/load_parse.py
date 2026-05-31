"""LoadAndParseStage — load a document from a file (or use given content)."""

from __future__ import annotations

import uuid
from typing import Any

from app.domains.ingestion.pipeline import MapperStage


class LoadAndParseStage(MapperStage):
    """If ``metadata['source_path']`` is set, load and parse the file; otherwise treat
    the incoming content as already loaded. Assigns a provisional ``doc_id`` (the
    DocumentResultSink overwrites it with the stored id)."""

    def __init__(self, supported_types: list[str] | None = None, **config: Any):
        types = supported_types or ["txt", "pdf", "html"]
        super().__init__(name="LoadAndParse", supported_types=types, **config)
        self.supported_types = types

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        source_path = metadata.get("source_path")
        content = self._load_file(source_path) if source_path else text
        return content, {"doc_id": metadata.get("doc_id") or str(uuid.uuid4()), "loaded": True}

    def _load_file(self, path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("txt", "text", "md"):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        if ext == "pdf":
            from pypdf import PdfReader

            return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        if ext in ("html", "htm"):
            import html2text

            with open(path, encoding="utf-8", errors="replace") as f:
                return html2text.html2text(f.read())
        raise ValueError(f"Unsupported file type: {ext!r}")

    def validate(self) -> None:
        return None
