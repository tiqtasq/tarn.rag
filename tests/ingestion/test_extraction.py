"""The built-in extractors: a Source -> StructuredDocument. Routing is the stage's job (test_load_parse)."""

import importlib.util

import pytest

from tarnrag.contracts import ElementKind
from tarnrag.ingestion.components.extraction import (
    HtmlExtractor,
    MarkdownExtractor,
    PlainTextExtractor,
    Source,
    TableJsonExtractor,
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


def test_markdown_lists():
    md = "# Gear\n\n- Helmet\n- Gloves\n- Boots\n"
    doc = _run(MarkdownExtractor, Source(source_id="d1", source_kind="markdown", content=md))
    items = [e for e in doc.elements if e.kind == ElementKind.LIST_ITEM]
    assert [e.text for e in items] == ["Helmet", "Gloves", "Boots"]
    assert all(e.header_path == ["Gear"] for e in items)  # under the heading
    for e in items:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text  # offset invariant holds for list items


def test_markdown_pipe_table():
    md = "# Specs\n\n| Bolt | Torque |\n| --- | --- |\n| M6 | 6 Nm |\n| M8 | 10 Nm |\n"
    doc = _run(MarkdownExtractor, Source(source_id="d1", source_kind="markdown", content=md))
    table_el = next(e for e in doc.elements if e.kind == ElementKind.TABLE)
    t = table_el.table
    assert t is not None and (t.n_rows, t.n_cols) == (3, 2)
    assert table_el.header_path == ["Specs"]
    cell = t.cell_at(1, 1)  # first data row, torque column
    assert cell.text == "6 Nm"
    assert [c.text for c in t.column_headers_for(cell)] == ["Torque"]  # query by column header
    s = table_el.geometry[0]
    assert doc.text[s.start:s.end] == table_el.text  # the table block indexes doc.text exactly


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


@pytest.mark.skipif(
    importlib.util.find_spec("bs4") is None, reason="beautifulsoup4 (parsers extra) not installed"
)
def test_html_headings_lists_and_table():
    html = (
        "<body><h1>Specs</h1><p>Torque values.</p>"
        "<ul><li>Helmet</li><li>Gloves</li></ul>"
        "<table><tr><th>Bolt</th><th>Torque</th></tr><tr><td>M6</td><td>6 Nm</td></tr></table>"
        "</body>"
    )
    doc = _run(HtmlExtractor, Source(source_id="d1", source_kind="html", content=html))
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.HEADING, "Specs") in kinds
    assert (ElementKind.LIST_ITEM, "Helmet") in kinds and (ElementKind.LIST_ITEM, "Gloves") in kinds
    assert next(e for e in doc.elements if e.kind == ElementKind.PARAGRAPH).header_path == ["Specs"]
    table_el = next(e for e in doc.elements if e.kind == ElementKind.TABLE)
    t = table_el.table
    assert (t.n_rows, t.n_cols) == (2, 2)
    cell = t.cell_at(1, 1)
    assert cell.text == "6 Nm"
    assert [c.text for c in t.column_headers_for(cell)] == ["Torque"]  # <th> in row 0 -> column header
    for e in doc.elements:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text  # offset invariant holds across all element kinds


@pytest.mark.skipif(
    importlib.util.find_spec("bs4") is None, reason="beautifulsoup4 (parsers extra) not installed"
)
def test_html_loose_text_skip_tags_code_and_nesting():
    """The remaining DOM-walk branches: loose text nodes -> paragraphs, skipped non-content subtrees
    (script/style), <pre> -> code, and recursion into container tags (<div>/<section>)."""
    html = (
        "<body>"
        "<style>.x{color:red}</style><script>var a = 1;</script>"  # skipped subtrees
        "Loose intro text."  # bare text node -> paragraph
        "<div><section><p>Nested para.</p></section></div>"  # recursion into containers
        "<pre>def f():\n    return 1</pre>"  # <pre> -> code (newlines preserved)
        "</body>"
    )
    doc = _run(HtmlExtractor, Source(source_id="d1", source_kind="html", content=html))
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.PARAGRAPH, "Loose intro text.") in kinds  # bare NavigableString captured
    assert (ElementKind.PARAGRAPH, "Nested para.") in kinds  # found by recursing into div/section
    assert (ElementKind.CODE, "def f():\n    return 1") in kinds  # <pre> block
    assert "var a = 1" not in doc.text and "color:red" not in doc.text  # script/style dropped


@pytest.mark.skipif(
    importlib.util.find_spec("bs4") is None, reason="beautifulsoup4 (parsers extra) not installed"
)
def test_html_empty_table_is_dropped():
    """A <table> with no rows renders to empty markdown (no cells) and produces no table element."""
    doc = _run(
        HtmlExtractor,
        Source(source_id="d1", source_kind="html", content="<body><table></table><p>after</p></body>"),
    )
    assert not any(e.kind == ElementKind.TABLE for e in doc.elements)
    assert any(e.text == "after" for e in doc.elements)


# ---------------- table_json: a JSON grid -> one atomic TABLE element with a real Table ----------------


def test_table_json_builds_a_native_table_element():
    """A bare JSON grid extracts to one TABLE element: cells with grid positions + header flags
    (defaults: first row = column headers, first column = row labels), markdown = the ` | `-joined grid
    (the stored/BM25 text), and the element is atomic."""
    import json

    grid = [["", "2019", "2018"], ["Goodwill", "1,910", "2,130"]]
    doc = _run(TableJsonExtractor, Source(source_id="d1", source_kind="table", content=json.dumps(grid)))
    [el] = doc.elements
    assert el.kind == ElementKind.TABLE and el.atomic
    assert doc.text == el.text == " | 2019 | 2018\nGoodwill | 1,910 | 2,130"  # exact cell tokens per row
    table = el.table
    assert (table.n_rows, table.n_cols) == (2, 3)
    value = table.cell_at(1, 1)
    assert value.text == "1,910"
    assert [h.text for h in table.column_headers_for(value)] == ["2019"]
    assert [h.text for h in table.row_headers_for(value)] == ["Goodwill"]


def test_table_json_object_form_overrides_header_shape():
    """The object form carries per-document header shape: header_rows=2 stacks column headers;
    header_cols=0 means no row labels."""
    import json

    content = json.dumps({"grid": [["FY", "FY"], ["2019", "2018"], ["10", "8"]], "header_rows": 2, "header_cols": 0})
    doc = _run(TableJsonExtractor, Source(source_id="d1", content=content))
    table = doc.elements[0].table
    value = table.cell_at(2, 0)
    assert [h.text for h in table.column_headers_for(value)] == ["FY", "2019"]  # stacked headers
    assert table.row_headers_for(value) == []  # header_cols=0 -> no row labels


def test_table_json_rejects_a_non_grid():
    import pytest

    with pytest.raises(ValueError, match="JSON grid"):
        _run(TableJsonExtractor, Source(source_id="d1", content='{"grid": "not a grid"}'))


def test_table_json_empty_grid_yields_no_elements():
    doc = _run(TableJsonExtractor, Source(source_id="d1", content="[]"))
    assert doc.elements == [] and doc.text == ""
