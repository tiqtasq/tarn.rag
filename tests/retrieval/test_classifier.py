"""The QueryClassifier seam: the NoOp default + the domain-independent StructuralQueryClassifier."""

import pytest

from tarnrag.retrieval import NoOpQueryClassifier, Query, StructuralQueryClassifier


def _structural(**overrides) -> StructuralQueryClassifier:
    return StructuralQueryClassifier(StructuralQueryClassifier.Config(**overrides))


def _classify(text: str, **overrides) -> Query:
    q = Query(text=text)
    _structural(**overrides).classify(q, None)  # the structural classifier ignores ctx
    return q


def test_noop_classifies_nothing():
    q = Query(text="anything at all")
    NoOpQueryClassifier(NoOpQueryClassifier.Config()).classify(q, None)
    assert q.query_type == "" and q.annotations == []


@pytest.mark.parametrize(
    "text",
    [
        "How do I keep a big metal container from rusting over time?",  # leading question word
        "Which marsupial is famous for grinning?",
        "is the vessel rated for that pressure?",  # trailing ? + auxiliary-led
    ],
)
def test_interrogative_is_semantic(text):
    assert _classify(text).query_type == "semantic"


@pytest.mark.parametrize(
    "text",
    [
        "shell thickness ultrasonic testing",  # content words only, no function words
        "mechanical seal bearing lubrication shaft alignment",
        "quokka",  # single keyword
    ],
)
def test_keyword_phrase_is_lexical(text):
    assert _classify(text).query_type == "lexical"


def test_function_word_heavy_phrase_is_semantic_without_a_question():
    # No question word, no '?', but dense with function words -> reads as natural language.
    q = _classify("the inspection of the tank over the years")
    assert q.query_type == "semantic"


@pytest.mark.parametrize(
    "text",
    [
        "wall thickness per ASME B31.3",  # alnum identifier
        "clause §6.4 acceptance criteria",  # section reference
        "tolerance for the 1.2.3 revision",  # dotted version (overrides the function words)
        'find the "exact phrase" in the manual',  # a quoted span
        "API 510 inspection",  # all-caps acronym + number
    ],
)
def test_exact_match_cues_force_lexical(text):
    # Even an otherwise natural-language phrasing routes lexical when it carries an exact-match cue.
    assert _classify(text).query_type == "lexical"


def test_records_a_rich_annotation():
    q = _classify("clause §6.4 acceptance criteria")
    assert len(q.annotations) == 1
    ann = q.annotations[0]
    assert ann.producer == "structural" and ann.type == "query_classification"
    assert ann.deterministic is True  # a heuristic, not generated
    assert ann.value["label"] == "lexical"
    assert ann.value["identifiers"] == ["§6.4"]
    assert ann.value["n_tokens"] == 4 and "function_word_ratio" in ann.value


def test_labels_and_threshold_are_configurable():
    # Rename the route keys and lower the threshold; "rate of the tank" (1 stopword / 4 -> 0.25) flips.
    q = _classify("rate of the tank", lexical_type="kw", semantic_type="nl", function_word_threshold=0.2)
    assert q.query_type == "nl"  # 0.25 >= 0.2 -> the semantic label (renamed)
    assert _classify("rate of the tank", function_word_threshold=0.6).query_type == "lexical"


def test_supplied_query_type_can_be_overwritten_by_the_classifier():
    # The classifier is authoritative when invoked directly; the router is what guards "supplied wins".
    q = Query(text="shell thickness ultrasonic testing", query_type="semantic")
    _structural().classify(q, None)
    assert q.query_type == "lexical"
