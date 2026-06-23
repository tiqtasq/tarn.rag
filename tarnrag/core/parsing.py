"""Shared parsing helpers — LLM replies follow strict-JSON conventions across layers.

Lives in ``core`` (not ``generation``) so both the generation components *and* the retrieval bridge
components (multi-query expansion, the LLM relevance judge) can parse strict-JSON replies without
``retrieval`` importing ``generation`` (the one-way ``generation → retrieval`` rule stays intact)."""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    """The first parseable JSON value in ``text`` — the whole string, else the outermost ``{...}`` (so a
    reply tolerates stray prose around the object). Returns ``None`` if neither parses."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
        return None
