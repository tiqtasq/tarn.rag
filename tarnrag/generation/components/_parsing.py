"""Shared parsing helpers for the generation components — re-exported from ``core`` (the single home, so
the retrieval bridge components can share it without ``retrieval`` importing ``generation``)."""

from __future__ import annotations

from tarnrag.core.parsing import extract_json

__all__ = ["extract_json"]
