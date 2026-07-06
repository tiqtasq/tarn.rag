"""The component framework moved to the standalone ``bausatz`` package; this shim keeps every
``tarnrag.core.components`` import working. The framework's own tests live in the bausatz repo —
here we only pin the re-export surface and that tarn.rag's registrations flow through it."""

from tarnrag.core.components import (
    Component,
    ComponentFactory,
    DuplicateTagError,
    Registry,
    UnknownTagError,
)


def test_shim_reexports_the_bausatz_framework():
    import bausatz
    import bausatz.registry

    assert Component is bausatz.Component
    assert ComponentFactory is bausatz.ComponentFactory
    assert Registry is bausatz.Registry
    assert DuplicateTagError is bausatz.registry.DuplicateTagError
    assert UnknownTagError is bausatz.registry.UnknownTagError


def test_tarnrag_components_register_through_the_shim():
    import tarnrag.ingestion  # noqa: F401 - importing registers the built-in stages/components
    import tarnrag.retrieval  # noqa: F401

    factory = ComponentFactory.get()
    for tag in ("pipeline", "retrieval_pipeline", "dense", "rrf", "structure_aware"):
        assert tag in factory.registry  # the domain registers into bausatz's global factory
