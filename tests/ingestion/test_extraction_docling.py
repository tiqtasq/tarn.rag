"""The Docling PDF extractor. The DoclingDocument → StructuredDocument MAPPING is verified against a
constructed DoclingDocument (needs only ``docling-core``); the end-to-end converter path is gated on
the heavy ``docling`` package being installed."""

import importlib.util
from pathlib import Path

import pytest

from tarnrag.contracts import ElementKind, StructuredDocument
from tarnrag.ingestion.extraction import Source
from tarnrag.ingestion.extraction.docling_pdf import DoclingExtractor

requires_docling_core = pytest.mark.skipif(
    importlib.util.find_spec("docling_core") is None, reason="docling-core not installed"
)


def _prov(page, box, span=(0, 0)):
    from docling_core.types.doc.base import CoordOrigin
    from docling_core.types.doc.document import BoundingBox, ProvenanceItem

    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
        charspan=span,
    )


@requires_docling_core
def test_mapping_headings_paragraph_geometry_and_header_path():
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    d = DoclingDocument(name="t")
    d.add_title(text="Pump Manual", prov=_prov(1, (10, 20, 200, 35)))
    h = d.add_heading(text="Safety", level=1, prov=_prov(1, (10, 40, 80, 55)))
    d.add_text(label=DocItemLabel.TEXT, text="Wear PPE.", parent=h, prov=_prov(1, (10, 60, 150, 75)))

    doc = DoclingExtractor._map(d, Source(source_id="d1", source_kind="pdf"))
    assert isinstance(doc, StructuredDocument) and doc.extractor == "docling"
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.HEADING, "Pump Manual") in kinds
    assert (ElementKind.HEADING, "Safety") in kinds
    assert (ElementKind.PARAGRAPH, "Wear PPE.") in kinds

    para = next(e for e in doc.elements if e.text == "Wear PPE.")
    assert para.header_path == ["Pump Manual", "Safety"]  # title (level 0) -> heading (level 1)
    box = para.geometry[0].boxes[0]
    assert box.page == 1 and box.bbox[2] > box.bbox[0]  # page + a real bbox
    s = para.geometry[0]
    assert doc.text[s.start:s.end] == "Wear PPE."  # char offsets index doc.text exactly


@requires_docling_core
def test_mapping_table_cells_geometry_and_header_addressing():
    from docling_core.types.doc.document import (
        DoclingDocument,
        ProvenanceItem,
        TableCell,
        TableData,
    )
    from docling_core.types.doc.base import BoundingBox, CoordOrigin

    def bb(box):
        return BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT)

    def cell(text, r, c, **kw):
        return TableCell(
            text=text, start_row_offset_idx=r, end_row_offset_idx=r + 1,
            start_col_offset_idx=c, end_col_offset_idx=c + 1, row_span=1, col_span=1, **kw,
        )

    data = TableData(num_rows=2, num_cols=2, table_cells=[
        cell("Bolt", 0, 0, column_header=True, bbox=bb((0, 0, 50, 10))),
        cell("Torque", 0, 1, column_header=True, bbox=bb((50, 0, 100, 10))),
        cell("M6", 1, 0, row_header=True, bbox=bb((0, 10, 50, 20))),
        cell("6 Nm", 1, 1, bbox=bb((50, 10, 100, 20))),
    ])
    d = DoclingDocument(name="t")
    d.add_table(data=data, prov=ProvenanceItem(
        page_no=3, bbox=bb((0, 0, 100, 20)), charspan=(0, 0)))

    doc = DoclingExtractor._map(d, Source(source_id="d1", source_kind="pdf"))
    table_el = next(e for e in doc.elements if e.kind == ElementKind.TABLE)
    t = table_el.table
    assert t is not None
    data_cell = t.cell_at(1, 1)
    assert data_cell.text == "6 Nm"
    assert data_cell.geometry[0].boxes[0].page == 3  # cell-level geometry off the table's page
    assert [c.text for c in t.column_headers_for(data_cell)] == ["Torque"]  # address by column header
    assert [c.text for c in t.row_headers_for(data_cell)] == ["M6"]  # address by row header
    s = data_cell.geometry[0]
    assert doc.text[s.start:s.end] == "6 Nm"  # cell char-offset indexes doc.text exactly


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None, reason="docling (the converter) not installed"
)
def test_docling_end_to_end_on_fixture():
    from tarnrag.ingestion.extraction import extract

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"
    doc = extract(Source(source_id="d1", source_kind="pdf", path=str(fixture),
                         metadata={"extractor": "docling"}))
    assert doc.extractor == "docling" and doc.elements


def test_docling_extractor_requires_a_path():
    with pytest.raises(ValueError):
        DoclingExtractor(DoclingExtractor.Config()).extract(Source(source_id="d1", content="x"))
