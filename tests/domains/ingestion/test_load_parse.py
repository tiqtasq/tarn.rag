"""LoadAndParseStage: per-request PDF backend selection via metadata['parser']."""

import pytest

from app.domains.base.models import PipelineItem
from app.domains.ingestion.stages.load_parse import LoadAndParseStage


def _stage():
    return LoadAndParseStage(
        pdf_parsers={"a": lambda p: f"A:{p}", "b": lambda p: f"B:{p}"},
        default_pdf_parser="a",
    )


def _run(stage, path, parser=None):
    meta = {"source_path": path}
    if parser is not None:
        meta["parser"] = parser
    [out] = list(stage.process(PipelineItem(content="", metadata=meta)))
    return out.content


def test_metadata_parser_selects_the_backend(tmp_path):
    pdf = str(tmp_path / "doc.pdf")
    assert _run(_stage(), pdf, parser="b") == f"B:{pdf}"


def test_absent_parser_uses_the_default(tmp_path):
    pdf = str(tmp_path / "doc.pdf")
    assert _run(_stage(), pdf) == f"A:{pdf}"


def test_unknown_parser_raises_listing_available(tmp_path):
    pdf = str(tmp_path / "doc.pdf")
    with pytest.raises(ValueError, match="Unknown pdf parser 'nope'"):
        _run(_stage(), pdf, parser="nope")


def test_txt_ignores_parser_and_reads_text(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")
    assert _run(_stage(), str(f), parser="b") == "hello world"


def test_invalid_default_parser_fails_validation():
    with pytest.raises(ValueError, match="default_pdf_parser"):
        LoadAndParseStage(pdf_parsers={"a": lambda p: p}, default_pdf_parser="missing")


def test_default_registry_has_pypdf_and_pdfplumber():
    stage = LoadAndParseStage()
    assert {"pypdf", "pdfplumber"} <= set(stage.pdf_parsers)
    assert stage.default_pdf_parser == "pypdf"


def test_real_pypdf_loader_extracts_text(tmp_path):
    """Smoke test the real pypdf backend end-to-end on a generated PDF (skipped if the
    optional deps aren't installed)."""
    pytest.importorskip("pypdf")
    rl = pytest.importorskip("reportlab.pdfgen.canvas", reason="reportlab needed to build a PDF")

    pdf = tmp_path / "real.pdf"
    c = rl.Canvas(str(pdf))
    c.drawString(72, 720, "hello pdf world")
    c.save()

    text = _run(LoadAndParseStage(), str(pdf))  # default = pypdf
    assert "hello pdf world" in text
