"""AcronymEnricher — a deterministic example enricher: tag ALL-CAPS tokens as entity annotations.

A minimal, dependency-free span-level enricher (PPE, OSHA, …): useful on its own and a template for
plugging real NER / topic enrichers behind the same ``Enricher`` port.
"""

from __future__ import annotations

import re
from typing import Literal

from tarnrag.contracts import Annotation, StructuredDocument
from tarnrag.ingestion.enrichment.enricher import Enricher

_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")  # 2+ consecutive capitals — an acronym/initialism token


class AcronymEnricher(Enricher):
    """Tag ALL-CAPS tokens (≥2 letters) as ``entity`` annotations with their char span (deterministic)."""

    class Config(Enricher.Config):
        class_name: Literal["acronyms"] = "acronyms"

    config: AcronymEnricher.Config

    def enrich(self, document: StructuredDocument) -> None:
        for element in document.elements:
            for match in _ACRONYM.finditer(element.text):
                element.annotations.append(
                    Annotation(
                        producer=self.config.class_name,
                        type="entity",
                        value={"text": match.group(), "kind": "acronym"},
                        span=[self._subspan(element, match.start(), match.end())],
                        deterministic=True,
                    )
                )
