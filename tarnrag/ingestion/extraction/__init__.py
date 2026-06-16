"""Layout-aware extraction: a ``Source`` → ``StructuredDocument`` seam + the built-in extractors.

Importing this package registers the built-in extractors with the global ``ComponentFactory`` (each
self-registers under its ``class_name``), so ``extract`` / ``ComponentFactory`` can build them.
"""

from tarnrag.ingestion.extraction.extractor import (
    DEFAULT_ROUTES,
    Extractor,
    Source,
    extract,
)
from tarnrag.ingestion.extraction.markdown import MarkdownExtractor
from tarnrag.ingestion.extraction.pdf import PdfTextExtractor
from tarnrag.ingestion.extraction.plain_text import PlainTextExtractor

__all__ = [
    "Extractor",
    "Source",
    "extract",
    "DEFAULT_ROUTES",
    "MarkdownExtractor",
    "PdfTextExtractor",
    "PlainTextExtractor",
]
