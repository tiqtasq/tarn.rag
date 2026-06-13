"""Retrieval public types (ModusQ spec §5.1 / §5.8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Purpose(str, Enum):
    EXECUTION = "EXECUTION"
    AUTHORING = "AUTHORING"
    GENERATION_GROUNDING = "GENERATION_GROUNDING"


@dataclass(frozen=True)
class MethodRef:
    method_id: str
    method_version: str | None = None  # None → latest in the index


# Sentinel for "no scope restriction" (whole index).
ALL = "ALL"


@dataclass
class Query:
    text: str
    purpose: Purpose = Purpose.EXECUTION
    scope: list[MethodRef] | str = ALL  # MethodRef[] or ALL
    top_k: int = 8
    dense_k: int = 50
    sparse_k: int = 50  # used in Step B (sparse retriever)


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    component_scores: dict[str, float]
    document_id: str
    source_kind: str
    standard_id: str | None
    locator: str | None
    license_class: str
    methods: list[MethodRef] = field(default_factory=list)
