"""Tag-dispatched factory that turns a serialized spec dict into a ``Component``.

The framework is three small modules, each with one job:

  Registry          (registry.py)         the tag -> class mapping, a pure lookup table
  Component         (component.py)        base for buildable classes; declares a typed nested
                                          Config and self-registers by its ``class_name`` tag
  ComponentFactory  (this module)         validate a spec into its Config, construct the
                                          Component, recursing into nested children

A ``ComponentFactory`` holds a ``Registry`` and is also reachable as a process-global singleton
via ``ComponentFactory.get()`` — which is where ``Component`` subclasses self-register on import.
``Component`` only references ``ComponentFactory`` inside a ``TYPE_CHECKING`` annotation (plus a
lazy import in ``__init_subclass__``), which avoids the import cycle between the two modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypeVar, Union

from pydantic import Field, TypeAdapter

from .registry import Registry
from .component import Component

_C = TypeVar("_C", bound=Component)


# -------------------------------------------------------------------------------------- #
# ComponentFactory: construction. Holds a Registry; turns specs into Component objects.  #
# -------------------------------------------------------------------------------------- #
class ComponentFactory:

    @staticmethod
    def get() -> ComponentFactory:
        return _component_factory

    def __init__(self, registry: Registry[Component] | None = None) -> None:
        self.registry: Registry[Component] = registry if registry is not None else Registry[Component]()

        self._adapter_cache: tuple[int, TypeAdapter[Any]] | None = None

    def create(self, spec: Mapping[str, Any]) -> Component:
        if not isinstance(spec, Mapping) or "class_name" not in spec:
            raise ValueError(f"spec must be a mapping with a 'class_name' key: {spec!r}")
        klass = self.registry.get(spec["class_name"])
        obj = klass(klass.Config.model_validate(dict(spec)))
        obj._ensure_children(self)  # build the tree against THIS factory + mark built (one owner)
        return obj

    def create_as(self, spec: Mapping[str, Any], expected: type[_C]) -> _C:
        """Build a component and assert it is an ``expected`` instance — a typed ``create`` for
        container components validating their children (e.g. a ``Chunk`` stage's chunker)."""
        obj = self.create(spec)
        if not isinstance(obj, expected):
            raise TypeError(
                f"{spec.get('class_name')!r} is a {type(obj).__name__}, not a {expected.__name__}"
            )
        return obj

    def create_many(self, specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Component]:
        """
        Build a dict of named layers, e.g. {"encoder": {...}, "head": {...}}.
        """
        return {name: self.create(spec) for name, spec in specs.items()}

    def validate(self, spec: Mapping[str, Any]) -> Component.Config:
        """
        Validate a spec into its Config without constructing the Component.
        """
        klass = self.registry.get(spec["class_name"])
        return klass.Config.model_validate(dict(spec))

    def config_adapter(self) -> TypeAdapter[Any]:
        """
        A Pydantic discriminated union over every registered Config.

        Validates a whole nested structure in one pass, with Pydantic routing each
        node by `class_name`. Rebuilt only when the registry's size changes.
        """
        if self._adapter_cache is not None and self._adapter_cache[0] == len(self.registry):
            return self._adapter_cache[1]
        configs = [cls.Config for cls in self.registry.classes()]
        if not configs:
            raise ValueError("registry is empty; nothing to validate against")
        if len(configs) == 1:
            adapter: TypeAdapter[Any] = TypeAdapter(configs[0])
        else:
            union = Union[tuple(configs)]  # type: ignore[valid-type]
            adapter = TypeAdapter(Annotated[union, Field(discriminator="class_name")])
        self._adapter_cache = (len(self.registry), adapter)
        return adapter

    def register(self, tag: str, cls: type[Component]) -> None:
        self.registry.register(tag, cls)


_component_factory = ComponentFactory()
