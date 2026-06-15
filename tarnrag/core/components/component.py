"""

Component class is the base class for ckasses that can be instantiated from
configs. This base for buildable classes; declares a typed nested Config and
registers itself by reading its Config's `class_name` discriminator.

Every concrete component pins its tag as a Literal field on its Config:

    class MyComponent(Component):
        class Config(Component.Config):
            class_name: Literal["linear"] = "my_component"
            other_variables: ...

so the tag travels with the serialized config (model_dump round-trips it) and is
validated like any other field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from tarnrag.core.components import ComponentFactory

# --------------------------------------------------------------------------- #
# Layer: base for config-driven classes.
# --------------------------------------------------------------------------- #
class Component:

    class Config(BaseModel):
        name: str | None = None                     # shared field for every layer
        # Concrete subclasses pin the discriminator, e.g.:
        #     class_name: Literal["linear"] = "linear"

        model_config = ConfigDict(extra="forbid")  # reject unknown keys; inherited by subclasses

    # Narrowed in subclasses (`config: Linear.Config`) purely for type checkers.
    config: Component.Config

    def __init_subclass__(cls, **kwargs: Any) -> None:
        from tarnrag.core.components import ComponentFactory
        super().__init_subclass__(**kwargs)
        field = cls.Config.model_fields.get("class_name")
        if field is None:
            return  # no discriminator pinned -> abstract base, not registered
        if field.default is PydanticUndefined:
            tag = cls.__name__.lower()
            raise TypeError(
                f"{cls.__qualname__}.Config.class_name needs a default, e.g. "
                f'class_name: Literal["{tag}"] = "{tag}"'
            )
        ComponentFactory.get().register(field.default, cls)

    def __init__(self, config: Component.Config) -> None:
        self.config = config

    def _build_children(self, factory: ComponentFactory) -> None:
        """
        Hook for containers. No-op for leaf layers.

        Override to construct nested layers from `self.config` using the SAME
        factory, so the whole tree is built against one registry:

            def _build_children(self, factory):
                self.layers = [factory.create(s) for s in self.config.layers]
        """

    def to_json(self) -> dict[str, Any]:
        """Serialize back to the dict shape that LayerFactory.create consumes."""
        return self.config.model_dump()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"
