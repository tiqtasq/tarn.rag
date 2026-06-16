"""Layout-aware extraction: a ``Source`` → ``StructuredDocument`` seam, the built-in extractors, and
the ``LoadAndParse`` stage that drives them.

Importing this package registers the built-in extractors *and* the ``LoadAndParse`` stage with the
global ``ComponentFactory`` (each self-registers under its ``class_name``), so ``Pipeline.from_spec`` /
``ComponentFactory`` can build them. ``LoadAndParseStage`` is imported last: it pulls in the ``extractor``
submodule, which the lines above have already loaded.
"""

from tarnrag.ingestion.extraction.extractor import Extractor, Source
from tarnrag.ingestion.extraction.docling_pdf import DoclingExtractor
from tarnrag.ingestion.extraction.html import HtmlExtractor
from tarnrag.ingestion.extraction.markdown import MarkdownExtractor
from tarnrag.ingestion.extraction.pdf import PdfTextExtractor
from tarnrag.ingestion.extraction.plain_text import PlainTextExtractor
from tarnrag.ingestion.extraction.load_parse import LoadAndParseStage

__all__ = [
    "Extractor",
    "Source",
    "DoclingExtractor",
    "HtmlExtractor",
    "MarkdownExtractor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "LoadAndParseStage",
]
