"""tarn.rag builds on the standalone ``bausatz`` component framework (PyPI) — the framework's own
tests live in its repo; here we pin only tarn.rag's side of the contract: importing the domain
packages registers every built-in component into bausatz's global factory."""

from bausatz import ComponentFactory


def test_domain_components_register_into_the_bausatz_factory():
    import tarnrag.generation  # noqa: F401 - importing registers the built-ins
    import tarnrag.ingestion  # noqa: F401
    import tarnrag.retrieval  # noqa: F401

    factory = ComponentFactory.get()
    for tag in (
        "pipeline", "LoadAndParse", "Chunk", "Embed", "structure_aware", "table_json",
        "retrieval_pipeline", "dense", "sparse", "rrf", "cross_encoder", "intent",
        "generation_pipeline", "single_hop", "decomposition", "answerability",
    ):
        assert tag in factory.registry, tag
