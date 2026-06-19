"""Generation pipeline — the answer orchestrator.

``GenerationPipeline`` (the container the engine builds from the ``GENERATION_PIPELINE`` spec): reason
(retrieve + read) → assemble the proof tree → ``GenerationResult``. The seams it composes live in
``generation/components/``. Import from the module
(``from tarnrag.generation.pipeline.pipeline import GenerationPipeline``).
"""
