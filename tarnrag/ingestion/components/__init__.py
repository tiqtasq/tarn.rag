"""Ingestion component families — the pluggable stage Components.

The config-driven Components the ingestion ``Pipeline`` composes from ``Settings``: ``extraction`` (format
extractors → ``StructuredDocument``), ``enrichment`` (annotate the document), and ``chunking`` (split into
chunks). Each self-registers with the global ``ComponentFactory`` on import. The stages with no component
family of their own (clean/normalize, embed) live with the ``Pipeline`` under ``ingestion/pipeline/``.
"""
