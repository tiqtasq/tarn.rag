"""Generation components — the seams the pipeline composes.

The config-driven Components a ``GenerationPipeline`` runs: the ``Reasoner`` (retrieve + read → answer +
cited steps; ``SingleHopReasoner`` / multi-hop ``IterativeReasoner`` / ``DecompositionReasoner``), the
``EvidenceAssembler`` (cited indices → provenance-bearing
``Citation``s; ``ProvenanceAssembler``), and the optional ``GroundingChecker`` (verify each claim against
its cited evidence; ``HeuristicGroundingChecker`` / ``LLMGroundingChecker``, composed by the cascading
``CascadingGroundingChecker``). Each self-registers with the
global ``ComponentFactory`` on import. Import from the modules
(e.g. ``from tarnrag.generation.components.reasoner import SingleHopReasoner``).
"""
