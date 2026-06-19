"""Ingestion pipeline — the Pipeline orchestrator + its loose stages.

``Pipeline`` (the config-driven stage container the ``IngestionEngine`` runs) plus the two stages with no
component family of their own: ``clean_normalize`` (CleanAndNormalize) and ``embed`` (Embed). The stages
backed by a component family live with it under ``ingestion/components/`` (LoadAndParse in extraction,
Enrich in enrichment, Chunk in chunking). Import from the modules
(e.g. ``from tarnrag.ingestion.pipeline.pipeline import Pipeline``).
"""
