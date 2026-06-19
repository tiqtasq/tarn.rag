"""CleanAndNormalizeStage — normalize whitespace and strip control characters."""

from __future__ import annotations

import re
from typing import Any, Literal

from tarnrag.ingestion.pipeline.pipeline import MapperStage

_C0_CONTROL = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")


class CleanAndNormalizeStage(MapperStage):
    """
    Strip control characters and (optionally) collapse runs of whitespace before
    chunking.
    """

    class Config(MapperStage.Config):
        class_name: Literal["CleanAndNormalize"] = "CleanAndNormalize"
        collapse_whitespace: bool = True

    config: CleanAndNormalizeStage.Config

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = _C0_CONTROL.sub("", text.replace("\x00", ""))
        if self.config.collapse_whitespace:
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip(), {"cleaned": True}
