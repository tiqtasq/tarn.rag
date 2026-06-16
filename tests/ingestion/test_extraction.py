"""The built-in extractors: a Source -> StructuredDocument. Routing is the stage's job (test_load_parse)."""

import importlib.util

import pytest

from tarnrag.contracts import ElementKind
from tarnrag.ingestion.extraction import (
    HtmlExtractor,
    MarkdownExtractor,
    PlainTextExtractor,
    Source,
)


def _run(cls, source: Source):
    return cls(cls.Config()).extract(source)


def test_plain_text_is_one_block_with_full_span():
    doc = _run(PlainTextExtractor, Source(source_id="d1", source_kind="text", content="hello world"))
    assert doc.extractor == "plain_text" and doc.text == "hello world"
    el = doc.elements[0]
    assert el.kind == ElementKind.PARAGRAPH
    assert (el.geometry[0].start, el.geometry[0].end) == (0, len("hello world"))


def test_plain_text_empty_source_yields_no_elements():
    doc = _run(PlainTextExtractor, Source(source_id="d1", content=""))
    assert doc.text == "" and doc.elements == []


def test_markdown_headings_paragraphs_code():
    md = "# Safety\n\nWear PPE.\n\n## Lockout\n\nIsolate the source.\n\n```\ncode here\n```\n"
    doc = _run(MarkdownExtractor, Source(source_id="d1", source_kind="markdown", content=md))
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.HEADING, "Safety") in kinds
    assert (ElementKind.PARAGRAPH, "Isolate the source.") in kinds
    assert (ElementKind.CODE, "code here") in kinds


def test_markdown_header_path_and_exact_offsets():
    md = "# Safety\n\n## Lockout\n\nIsolate the source."
    doc = _run(MarkdownExtractor, Source(source_id="d1", source_kind="markdown", content=md))
    para = next(e for e in doc.elements if e.text == "Isolate the source.")
    assert para.header_path == ["Safety", "Lockout"]  # nested headings
    lockout = next(e for e in doc.elements if e.text == "Lockout")
    assert para.parent_id == lockout.id
    for e in doc.elements:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text  # offsets index doc.text exactly


def test_markdown_reads_from_path(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nHello.", encoding="utf-8")
    doc = _run(MarkdownExtractor, Source(source_id="d1", source_kind="md", path=str(p)))
    assert any(e.kind == ElementKind.HEADING and e.text == "Title" for e in doc.elements)


@pytest.mark.skipif(
    importlib.util.find_spec("bs4") is None, reason="beautifulsoup4 (parsers extra) not installed"
)
def test_html_strips_markup_to_text():
    doc = _run(HtmlExtractor, Source(source_id="d1", source_kind="html",
                                     content="<h1>Hi</h1><p>Body &amp; more.</p>"))
    assert doc.extractor == "html"
    assert "Hi" in doc.text and "Body & more." in doc.text and "<h1>" not in doc.text
