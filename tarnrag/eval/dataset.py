"""The labeled query set for the retrieval eval harness.

Relevance is **content-based**: a result is relevant iff its text contains one of the query's gold
``relevant`` phrases (case-insensitive). This is deliberate — chunk ids are not stable across ingestion
configs (they're fresh uuids), so id-based labels wouldn't survive re-ingestion. Content matching does,
which is what lets the harness compare **embed-time variants** (e.g. header-path injection, which rebuilds
the index) and not just retrieval-side pipeline specs against one fixed index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tarnrag.retrieval.types import Purpose


@dataclass
class EvalQuery:
    """One labeled query: the text, the gold ``relevant`` phrases (a result is relevant iff its text
    contains any, case-insensitive), and the retrieval ``purpose``."""

    text: str
    relevant: list[str]
    purpose: Purpose = Purpose.EXECUTION


@dataclass
class EvalSet:
    """A labeled query set — the ground truth a sweep is scored against."""

    queries: list[EvalQuery] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: list[dict]) -> EvalSet:
        """Build from plain dicts: ``{"text": ..., "relevant": [...], "purpose"?: "EXECUTION"}``."""
        return cls(
            [
                EvalQuery(
                    text=r["text"],
                    relevant=list(r["relevant"]),
                    purpose=Purpose(r.get("purpose", Purpose.EXECUTION)),
                )
                for r in records
            ]
        )

    @classmethod
    def from_json(cls, path: str | Path) -> EvalSet:
        """Load from a JSON file holding a list of query records (see :meth:`from_records`)."""
        return cls.from_records(json.loads(Path(path).read_text()))
