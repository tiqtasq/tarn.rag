"""Config-driven component framework: build classes from dict/JSON specs by a ``class_name`` tag.

A ``Component`` declares a typed nested ``Config`` and self-registers on import; ``ComponentFactory``
turns a spec into an instance; ``Registry`` is the tag -> class table. See ``component.py`` for the
registration mechanics.
"""

from tarnrag.core.components.registry import Registry
from tarnrag.core.components.component import Component
from tarnrag.core.components.component_factory import ComponentFactory

__all__ = ["Component", "ComponentFactory", "Registry"]
