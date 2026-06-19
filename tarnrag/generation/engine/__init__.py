"""Generation engine — the question-answering facade.

``GenerationEngine`` (``GenerationEngine.create()``) wires the retrieval port + the ``LanguageModel`` + the
``GenerationPipeline``, and answers a question with an evidence-bearing proof tree. Import from the module
(``from tarnrag.generation.engine.engine import GenerationEngine``).
"""
