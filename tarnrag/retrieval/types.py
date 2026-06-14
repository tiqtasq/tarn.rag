"""Retrieval query types (ModusQ spec §5.1). Result types live in ``tarnrag.contracts``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tarnrag.contracts import MethodRef


class Purpose(str, Enum):
    """
    Retrieval intent (ModusQ §5.1) — drives scope/licensing policy in later steps.
    """

    EXECUTION = "EXECUTION"
    AUTHORING = "AUTHORING"
    GENERATION_GROUNDING = "GENERATION_GROUNDING"


# Sentinel for "no scope restriction" (whole index).
ALL = "ALL"


@dataclass
class Query:
    """
    A retrieval request: the query text plus knobs (purpose, method scope, top_k / dense_k /
    sparse_k).
    """

    text: str
    purpose: Purpose = Purpose.EXECUTION
    scope: list[MethodRef] | str = ALL  # MethodRef[] or ALL
    top_k: int = 8
    dense_k: int = 50
    sparse_k: int = 50  # used in Step B (sparse retriever)
