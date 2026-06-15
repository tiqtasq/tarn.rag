"""Tag-dispatched factory for a family of config-driven classes.

Three pieces, each with one job:

  Registry      owns the tag -> class mapping (a pure lookup table)
  Layer         base for buildable classes; declares a typed nested Config and
                registers itself by reading its Config's `class_name` discriminator
  LayerFactory  turns a serialized spec dict into a Layer instance (validate then
                construct, recursing into nested children)


To split across modules: Registry and the exceptions go in one file, Layer in a
second, LayerFactory in a third. Layer references LayerFactory only inside an
annotation, so guard that import with `if TYPE_CHECKING:` to avoid an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import PydanticUndefined

from .registry import Registry
from .component import Component



# -------------------------------------------------------------------------------------- #
# ComponentFactory: construction. Holds a Registry; turns specs into Component objects.  #
# -------------------------------------------------------------------------------------- #
class ComponentFactory:

    @staticmethod
    def get() -> ComponentFactory:
        return _componentFactory

    def __init__(self, registry: Registry[Component] | None = None) -> None:
        self.registry: Registry[Component] = registry if registry is not None else Registry[Component]()

        self._adapter_cache: tuple[int, TypeAdapter[Any]] | None = None

    def create(self, spec: Mapping[str, Any]) -> Component:
        if not isinstance(spec, Mapping) or "class_name" not in spec:
            raise ValueError(f"spec must be a mapping with a 'class_name' key: {spec!r}")
        klass = self.registry.get(spec["class_name"])
        obj = klass(klass.Config.model_validate(dict(spec)))
        obj._build_children(self)
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


_componentFactory = ComponentFactory()
