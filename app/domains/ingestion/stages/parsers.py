"""PDF text-extraction backends — the strategy registry for ``LoadAndParseStage``.

Each loader is a pure ``(path) -> str`` function with its heavy dependency imported lazily
(so the SQLite/test path and the non-PDF flow stay dependency-light). The *set* of available
backends is stage configuration (built once); a request selects one per call via
``metadata['parser']`` (see ``LoadAndParseStage``). Add a new backend by writing a loader and
registering it here — OCR/cloud backends plug in the same way.
"""

from __future__ import annotations

from collections.abc import Callable


def load_pypdf(path: str) -> str:
    """Extract text with pypdf (BSD; pure-Python; good for simple born-digital PDFs)."""
    from pypdf import PdfReader

    return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)


def load_pdfplumber(path: str) -> str:
    """Extract text with pdfplumber (MIT; better layout/table handling, slower)."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


# Registry of PDF backends by name (the choice rides in metadata['parser']).
DEFAULT_PDF_PARSERS: dict[str, Callable[[str], str]] = {
    "pypdf": load_pypdf,
    "pdfplumber": load_pdfplumber,
}
DEFAULT_PDF_PARSER = "pypdf"
AVAILABLE_PDF_PARSERS = frozenset(DEFAULT_PDF_PARSERS)
