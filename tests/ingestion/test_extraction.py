"""The extraction seam: a Source becomes a StructuredDocument via registered Extractor Components,
with routing (override → per-kind default → plain-text fallback)."""

from tarnrag.contracts import ElementKind
from tarnrag.ingestion.extraction import Source, extract


def test_plain_text_is_one_block_with_full_span():
    doc = extract(Source(source_id="d1", source_kind="text", content="hello world"))
    assert doc.extractor == "plain_text"
    assert doc.text == "hello world"
    assert len(doc.elements) == 1
    el = doc.elements[0]
    assert el.kind == ElementKind.PARAGRAPH
    assert (el.geometry[0].start, el.geometry[0].end) == (0, len("hello world"))


def test_empty_source_yields_no_elements():
    doc = extract(Source(source_id="d1", source_kind="text", content=""))
    assert doc.text == "" and doc.elements == []


def test_unknown_kind_falls_back_to_plain_text():
    doc = extract(Source(source_id="d1", source_kind="rtf", content="x"))
    assert doc.extractor == "plain_text"  # graceful degradation (FR-1.3)


def test_explicit_extractor_override_wins():
    # force plain_text even on a markdown source via the routing hint
    doc = extract(Source(source_id="d1", source_kind="markdown", content="# H\n\nx",
                         metadata={"extractor": "plain_text"}))
    assert doc.extractor == "plain_text"
    assert len(doc.elements) == 1  # not split into heading + paragraph


def test_markdown_headings_paragraphs_code():
    md = "# Safety\n\nWear PPE.\n\n## Lockout\n\nIsolate the source.\n\n```\ncode here\n```\n"
    doc = extract(Source(source_id="d1", source_kind="markdown", content=md))
    assert doc.extractor == "markdown"
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.HEADING, "Safety") in kinds
    assert (ElementKind.HEADING, "Lockout") in kinds
    assert (ElementKind.PARAGRAPH, "Isolate the source.") in kinds
    assert (ElementKind.CODE, "code here") in kinds


def test_markdown_header_path_and_parent():
    md = "# Safety\n\n## Lockout\n\nIsolate the source."
    doc = extract(Source(source_id="d1", source_kind="markdown", content=md))
    para = next(e for e in doc.elements if e.text == "Isolate the source.")
    assert para.header_path == ["Safety", "Lockout"]  # nested headings
    lockout = next(e for e in doc.elements if e.text == "Lockout")
    assert para.parent_id == lockout.id  # parented to the deepest enclosing heading


def test_markdown_offsets_are_exact_into_normalized_text():
    md = "# T\n\nbody"
    doc = extract(Source(source_id="d1", source_kind="markdown", content=md))
    for e in doc.elements:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text  # geometry indexes StructuredDocument.text exactly


def test_extractor_reads_from_path(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nHello.", encoding="utf-8")
    doc = extract(Source(source_id="d1", source_kind="md", path=str(p)))
    assert doc.extractor == "markdown"
    assert any(e.kind == ElementKind.HEADING and e.text == "Title" for e in doc.elements)
