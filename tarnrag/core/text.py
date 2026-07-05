"""Deterministic query-text cues — the exact-match signals shared across layers.

The structural query classifier reads these cues to label lexical intent (a quoted span or an
identifier ⇒ ``query_type='lexical'``), and the sparse-query builders act on the same cues (a
quoted span or a punctuation-split identifier becomes a required phrase). One definition here
keeps the two from drifting: what the classifier calls an identifier is exactly what the FTS
builder phrases. LLM-free, deterministic, C++-portable.
"""

from __future__ import annotations

import re
import string

_QUOTED = re.compile(r'"([^"]+)"')
_VERSION = re.compile(r"\d+(?:[.\-]\d+)+")  # 6.4, 1.2.3, 2024-01 — dotted/hyphenated number runs


def quoted_spans(text: str) -> list[str]:
    """The double-quoted spans in ``text`` — explicit exact-match intent — quote-free, blank-free."""
    return [s for s in (m.strip() for m in _QUOTED.findall(text)) if s]


def looks_like_identifier(token: str) -> bool:
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
