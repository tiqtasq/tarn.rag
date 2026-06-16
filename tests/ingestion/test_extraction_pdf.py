"""The born-digital PDF extractor over a real fixture. Gated on pdfplumber (the optional 'parsers'
extra); auto-skipped when it isn't installed, so a bare CI stays green."""

import importlib.util
from pathlib import Path

import pytest

from tarnrag.ingestion.extraction import Source, extract

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pdfplumber") is None,
    reason="pdfplumber (the 'parsers' extra) is not installed",
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"


def test_pdf_routes_to_pdf_text_and_carries_page_bbox():
    doc = extract(Source(source_id="d1", source_kind="pdf", path=str(FIXTURE)))
    assert doc.extractor == "pdf_text"
    assert doc.elements, "a born-digital PDF should yield text elements"
    box = doc.elements[0].geometry[0].boxes[0]
    assert box.page == 1  # PDF geometry: page + bbox (highlight-grade)
    assert box.bbox[2] > box.bbox[0] and box.bbox[3] > box.bbox[1]  # a real, non-degenerate box


def test_pdf_offsets_index_the_normalized_text_exactly():
    # char offsets work on PDF too — the universal substrate, not just visual boxes
    doc = extract(Source(source_id="d1", source_kind="pdf", path=str(FIXTURE)))
    for e in doc.elements:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text


def test_pdf_text_extractor_requires_a_path():
    with pytest.raises(ValueError):
        extract(Source(source_id="d1", source_kind="pdf", content="not a pdf",
                       metadata={"extractor": "pdf_text"}))
