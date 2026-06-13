"""CleanAndNormalizeStage — normalize whitespace and strip control characters."""

from __future__ import annotations

import re
from typing import Any

from tarnrag.ingestion.pipeline import MapperStage

_C0_CONTROL = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")


class CleanAndNormalizeStage(MapperStage):
    """Strip control characters and (optionally) collapse runs of whitespace before
    chunking."""

    def __init__(self, collapse_whitespace: bool = True, **config: Any):
        super().__init__(
            name="CleanAndNormalize", collapse_whitespace=collapse_whitespace, **config
        )
        self.collapse_whitespace = collapse_whitespace

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = _C0_CONTROL.sub("", text.replace("\x00", ""))
        if self.collapse_whitespace:
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip(), {"cleaned": True}

    def validate(self) -> None:
        return None
