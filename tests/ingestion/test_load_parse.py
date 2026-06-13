"""LoadAndParseStage: per-request PDF backend selection via metadata['parser']."""

from pathlib import Path

import pytest

from tarnrag.storage.models import PipelineItem
from tarnrag.ingestion.stages.load_parse import LoadAndParseStage

# A tiny real PDF whose only text is "Quokka ingestion smoke 7" (committed fixture).
SAMPLE_PDF = Path(__file__).parents[1] / "fixtures" / "sample.pdf"


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


def test_html_is_extracted_with_tags_stripped(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<h1>Hello</h1><p>World &amp; more</p>", encoding="utf-8")
    out = _run(LoadAndParseStage(), str(f))  # real bs4 loader
    assert "Hello" in out and "World" in out
    assert "<h1>" not in out  # tags stripped
    assert "&" in out and "&amp;" not in out  # entities decoded


def test_invalid_default_parser_fails_validation():
    with pytest.raises(ValueError, match="default_pdf_parser"):
        LoadAndParseStage(pdf_parsers={"a": lambda p: p}, default_pdf_parser="missing")


def test_default_registry_has_pypdf_and_pdfplumber():
    stage = LoadAndParseStage()
    assert {"pypdf", "pdfplumber"} <= set(stage.pdf_parsers)
    assert stage.default_pdf_parser == "pypdf"


@pytest.mark.parametrize("parser", [None, "pypdf", "pdfplumber"])
def test_real_pdf_backends_extract_text(parser):
    """Each real backend extracts text from the committed sample PDF (None → default pypdf)."""
    text = _run(LoadAndParseStage(), str(SAMPLE_PDF), parser=parser)
    assert "Quokka" in text
