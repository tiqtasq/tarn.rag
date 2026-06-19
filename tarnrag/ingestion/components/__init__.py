"""Ingestion component families — the pluggable stage Components.

The config-driven Components the ingestion ``Pipeline`` composes from ``Settings``: ``extraction`` (format
extractors → ``StructuredDocument``), ``enrichment`` (annotate the document), and ``chunking`` (split into
chunks). Each self-registers with the global ``ComponentFactory`` on import. The non-pluggable stages
(clean/normalize, embed) live at the ``ingestion`` top level alongside the ``pipeline``.
"""
