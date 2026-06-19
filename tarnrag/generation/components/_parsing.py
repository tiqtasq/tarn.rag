"""Shared parsing helpers for the generation components — the LLM replies follow strict-JSON conventions."""

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
