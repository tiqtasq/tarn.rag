"""Config-driven component framework — now the standalone ``bausatz`` package (PyPI), re-exported
here so every existing ``tarnrag.core.components`` import keeps working unchanged.

The framework (``Component`` + typed nested pydantic ``Config`` + self-registration by ``class_name``
tag, ``ComponentFactory``'s recursive spec→instance construction, ``Registry``) was extracted to
https://pypi.org/project/bausatz/ — its docs and tests live there. tarn.rag registers into the
process-global factory (bausatz's default); the framework also supports scoped registries
(``class MyBase(Component, factory=...)``) should co-resident libraries ever need them.
"""

from bausatz import Component, ComponentFactory, Registry
from bausatz.registry import DuplicateTagError, UnknownTagError

__all__ = ["Component", "ComponentFactory", "Registry", "DuplicateTagError", "UnknownTagError"]
