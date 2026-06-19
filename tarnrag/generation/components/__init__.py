"""Generation components — the seams the pipeline composes.

The config-driven Components a ``GenerationPipeline`` runs: the ``Reasoner`` (retrieve + read → answer +
cited steps; ``SingleHopReasoner``) and the ``EvidenceAssembler`` (cited indices → provenance-bearing
``Citation``s; ``ProvenanceAssembler``). Each self-registers with the global ``ComponentFactory`` on
import. Import from the modules (e.g. ``from tarnrag.generation.components.reasoner import SingleHopReasoner``).
"""
