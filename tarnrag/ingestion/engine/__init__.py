"""Ingestion engine + run-time — the producer facade and the distributed-execution machinery.

``IngestionEngine`` (the producer/query facade) plus the run-time it drives: the ``orchestrator`` (pipeline
execution), ``jobs`` (job records / status transitions), and the ``worker`` (the distributed-mode consumer
entry point, ``run_worker``). Import from the modules
(e.g. ``from tarnrag.ingestion.engine.engine import IngestionEngine``).
"""
