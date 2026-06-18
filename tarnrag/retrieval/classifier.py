"""The QueryClassifier seam — populate a query's classification (``query_type`` + ``annotations``).

A ``QueryClassifier`` is a config-driven ``Component`` that writes its findings onto the ``Query`` the
same way an ``Enricher`` annotates a chunk: it sets the cheap ``query_type`` route key and appends rich
``Annotation``s (each carrying the ``deterministic`` flag, so an LLM classifier's findings are flagged,
never silently trusted). Classification is opt-in:

- ``NoOpQueryClassifier`` — the **default**; classifies nothing (an unconfigured router falls through to
  its ``default`` route ≡ a single pipeline).
- ``StructuralQueryClassifier`` — the first real, **domain-independent** classifier: a heuristic over the
  query's *form* (not its subject), → ``query_type ∈ {lexical, semantic}``.

The ``RoutingRetrievalPipeline`` runs the configured classifier before dispatching on ``query.query_type``;
downstream consumers (the eval harness, logging, later the ``Reasoner``) read the same fields off the
contract. Domain-specific classifiers are added later as more components on this seam — no base edits.
"""

from __future__ import annotations

import re
import string
from abc import abstractmethod
from typing import Any, Literal

from tarnrag.contracts import Annotation, Geometry
from tarnrag.core.components import Component
from tarnrag.retrieval.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class QueryClassifier(Component):
    """Port: classify a ``Query`` in place — set ``query.query_type`` and/or append ``query.annotations``.
    Subclasses call ``annotate`` rather than touching ``.annotations`` directly, so the ``producer`` is
    recorded automatically (mirrors ``Enricher.annotate``)."""

    class Config(Component.Config):
        """Base classifier config; concrete classifiers pin ``class_name``."""

    @abstractmethod
    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        """Annotate ``query`` in place: set its ``query_type`` and/or append annotations. ``ctx`` is
        available for classifiers that need the embedder/store/LLM (the structural one ignores it)."""

    def annotate(
        self,
        query: Query,
        type: str,
        value: dict[str, Any],
        *,
        span: Geometry | None = None,
        deterministic: bool = True,
    ) -> None:
        """Append one finding by this classifier to ``query.annotations``; ``producer`` is filled in.
        ``span`` may mark the sub-region of the query text it applies to (char offsets into ``query.text``)."""
        query.annotations.append(
            Annotation(
                producer=self.config.class_name,
                type=type,
                value=value,
                span=span,
                deterministic=deterministic,
            )
        )


class NoOpQueryClassifier(QueryClassifier):
    """The default: classify nothing, leaving ``query_type`` as supplied. So a router with no classifier
    configured dispatches on a caller-supplied ``query_type`` (or its ``default`` route when none)."""

    class Config(QueryClassifier.Config):
        class_name: Literal["noop"] = "noop"

    config: NoOpQueryClassifier.Config

    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        return None


# Compact, domain-independent word lists (English defaults; override via Config). They lean English, but
# the exact-match cues (identifiers / quotes / length) are language-independent and carry weight regardless.
_QUESTION_WORDS = frozenset(
    "who what when where why how which whose whom is are was were do does did "
    "can could should would will may might has have had".split()
)
_STOPWORDS = frozenset(
    "a an the of to in on for and or but with at by from as into about than that this these those "
    "be been being it its i we you they he she my your our their over under up down out".split()
)
_VERSION = re.compile(r"\d+(?:[.\-]\d+)+")  # 6.4, 1.2.3, 2024-01 — dotted/hyphenated number runs


class StructuralQueryClassifier(QueryClassifier):
    """Domain-independent heuristic: classify a query by its *form* (keyword/exact vs natural-language)
    rather than its subject. Signals — interrogative form, function-word ratio, exact-match cues
    (double-quoted spans / identifiers / acronyms) — map to ``query_type ∈ {lexical, semantic}``, and the
    features found are recorded as an annotation. Deterministic, LLM-free, C++-portable;
    labels / threshold / word-lists are config. The decision is a short ordered rule, each clause
    defensible on its own:

    1. a quoted span or an identifier ⇒ ``lexical`` (explicit exact-match intent);
    2. otherwise an interrogative or function-word-heavy phrasing ⇒ ``semantic`` (natural language);
    3. otherwise (content words, no function words) ⇒ ``lexical`` (a keyword query).
    """

    class Config(QueryClassifier.Config):
        class_name: Literal["structural"] = "structural"
        lexical_type: str = "lexical"
        semantic_type: str = "semantic"
        function_word_threshold: float = 0.25  # fn-word ratio at/above which a phrase reads as semantic
        stopwords: list[str] | None = None  # None ⇒ built-in English set
        question_words: list[str] | None = None  # None ⇒ built-in English set

    config: StructuralQueryClassifier.Config

    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        cfg = self.config
        stop = frozenset(cfg.stopwords) if cfg.stopwords is not None else _STOPWORDS
        qwords = frozenset(cfg.question_words) if cfg.question_words is not None else _QUESTION_WORDS

        text = query.text.strip()
        raw_tokens = text.split()
        words = [w for w in (t.strip(string.punctuation).lower() for t in raw_tokens) if w]
        n = len(words)

        fn_ratio = sum(w in stop for w in words) / n if n else 0.0
        interrogative = bool(words) and (words[0] in qwords or text.endswith("?"))
        has_quotes = text.count('"') >= 2
        identifiers = [t for t in raw_tokens if self._looks_like_identifier(t)]

        if has_quotes or identifiers:
            label = cfg.lexical_type
        elif interrogative or fn_ratio >= cfg.function_word_threshold:
            label = cfg.semantic_type
        else:
            label = cfg.lexical_type

        query.query_type = label
        self.annotate(
            query,
            "query_classification",
            {
                "label": label,
                "interrogative": interrogative,
                "function_word_ratio": round(fn_ratio, 3),
                "n_tokens": n,
                "has_quotes": has_quotes,
                "identifiers": identifiers,
            },
        )

    @staticmethod
    def _looks_like_identifier(token: str) -> bool:
        """Whether a whitespace token reads as an identifier / code (a language-independent exact-match
        cue): a ``§`` reference, a letter+digit mix (``BM25``, ``L6``, ``v2``), an all-caps acronym
        (``API``, ``PDF``), or a dotted/hyphenated number (``6.4``, ``1.2.3``)."""
        if "§" in token:
            return True
        core = token.strip(string.punctuation)
        if not core:
            return False
        has_alpha = any(c.isalpha() for c in core)
        has_digit = any(c.isdigit() for c in core)
        if has_alpha and has_digit:
            return True
        if len(core) >= 2 and core.isupper() and has_alpha:
            return True
        return bool(_VERSION.fullmatch(core))
