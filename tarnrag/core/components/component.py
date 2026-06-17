"""Base class for config-driven classes — the ``Component`` family.

A ``Component`` declares a typed nested ``Config`` and self-registers by reading its Config's
``class_name`` discriminator, so it can be built from a plain dict/JSON spec via
``ComponentFactory``. Every concrete component pins its tag as a ``Literal`` field on its Config,
equal to the field's default:

    class MyComponent(Component):
        class Config(Component.Config):
            class_name: Literal["my_component"] = "my_component"
            ...other fields...

so the tag travels with the serialized config (``model_dump`` round-trips it) and is validated
like any other field. Registration happens at class-definition time into the process-global
``ComponentFactory``, so a component must be IMPORTED before the factory can build it from a spec.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from tarnrag.core.components import ComponentFactory

# --------------------------------------------------------------------------- #
# Component: base for config-driven classes.
# --------------------------------------------------------------------------- #
class Component(ABC):

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
        self._children_built = False  # flipped True by _ensure_children once the children are built

    def _build_children(self, factory: ComponentFactory) -> None:
        """
        Hook for container components. No-op for leaf components.

        Override to construct nested components from ``self.config`` using the SAME
        factory, so the whole tree is built against one registry:

            def _build_children(self, factory):
                self.children = [factory.create(spec) for spec in self.config.children]
        """

    def _ensure_children(self, factory: ComponentFactory | None = None) -> None:
        """Build the children once and mark them built — the single owner of both the build and the
        ``_children_built`` flag, so they can't drift apart. ``ComponentFactory.create`` calls this with
        itself, so the tree builds against the one registry; a directly-constructed container (tests,
        ``Pipeline.from_stages``) calls it with no factory before first use and falls back to the global
        one. Idempotent: a second call is a no-op."""
        if not self._children_built:
            from tarnrag.core.components import ComponentFactory

            self._children_built = True  # commit before building: a re-entrant call won't rebuild
            self._build_children(factory or ComponentFactory.get())

    def to_json(self) -> dict[str, Any]:
        """Serialize back to the dict shape that ``ComponentFactory.create`` consumes."""
        return self.config.model_dump()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"
