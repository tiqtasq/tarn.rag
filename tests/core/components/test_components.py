"""Component framework: self-registration, tag-dispatched construction,
recursive child building, config validation, and the discriminated-union adapter.

The Linear / Dropout / Sequential classes below are throwaway test doubles that
exercise the three component shapes: a leaf (Linear), a leaf with a constrained
field (Dropout), and a container that builds nested children (Sequential).
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import Field, ValidationError

from tarnrag.core.components import Component, ComponentFactory, Registry
from tarnrag.core.components.registry import DuplicateTagError, UnknownTagError


# --------------------------------------------------------------------------- #
# Test components (auto-register into the global factory on definition).
# --------------------------------------------------------------------------- #
class Linear(Component):
    class Config(Component.Config):
        class_name: Literal["linear"] = "linear"
        in_features: int
        out_features: int
        bias: bool = True

    config: Linear.Config

    def forward(self, x: float) -> float:
        return x * self.config.out_features + (1.0 if self.config.bias else 0.0)


class Dropout(Component):
    class Config(Component.Config):
        class_name: Literal["dropout"] = "dropout"
        p: float = Field(ge=0.0, le=1.0)

    config: Dropout.Config

    def forward(self, x: float) -> float:
        return x * (1.0 - self.config.p)


class Sequential(Component):
    class Config(Component.Config):
        class_name: Literal["sequential"] = "sequential"
        layers: list[dict[str, Any]]  # raw child specs, each with its own class_name

    config: Sequential.Config

    def __init__(self, config: Sequential.Config) -> None:
        super().__init__(config)
        self.layers: list[Component] = []

    def _build_children(self, factory: ComponentFactory) -> None:
        self.layers = [factory.create(spec) for spec in self.config.layers]

    def forward(self, x: float) -> float:
        for layer in self.layers:
            x = layer.forward(x)
        return x


@pytest.fixture
def factory() -> ComponentFactory:
    """A fresh, isolated factory with the three test components registered.

    Isolated from the process-global singleton so tag/adapter assertions stay
    deterministic regardless of what else the test session has registered.
    """
    f = ComponentFactory(Registry())
    f.register("linear", Linear)
    f.register("dropout", Dropout)
    f.register("sequential", Sequential)
    return f


# --------------------------------------------------------------------------- #
# Self-registration via __init_subclass__.
# --------------------------------------------------------------------------- #
def test_components_self_register_in_global_factory():
    reg = ComponentFactory.get().registry
    assert reg.get("linear") is Linear
    assert reg.get("dropout") is Dropout
    assert reg.get("sequential") is Sequential


def test_class_name_without_default_is_rejected():
    with pytest.raises(TypeError, match="needs a default"):

        class Bad(Component):
            class Config(Component.Config):
                class_name: Literal["bad"]  # no default -> cannot self-register


def test_component_without_discriminator_is_abstract():
    # No class_name pinned -> treated as an abstract base, silently not registered.
    class Abstract(Component):
        class Config(Component.Config):
            pass

    assert "class_name" not in Abstract.Config.model_fields


# --------------------------------------------------------------------------- #
# Construction: create / create_many, recursive children.
# --------------------------------------------------------------------------- #
def test_create_returns_typed_instance(factory):
    obj = factory.create(
        {"class_name": "linear", "name": "enc", "in_features": 128, "out_features": 64}
    )
    assert isinstance(obj, Linear)
    assert obj.config.name == "enc"
    assert obj.config.in_features == 128
    assert obj.config.bias is True  # default applied


def test_create_many_builds_named_components(factory):
    config = {
        "encoder": {
            "class_name": "linear",
            "name": "enc",
            "in_features": 128,
            "out_features": 64,
        },
        "regularizer": {"class_name": "dropout", "p": 0.5},
        "head": {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        },
    }
    built = factory.create_many(config)

    assert set(built) == {"encoder", "regularizer", "head"}
    assert isinstance(built["encoder"], Linear)
    assert isinstance(built["regularizer"], Dropout)
    assert isinstance(built["head"], Sequential)


def test_container_builds_nested_children(factory):
    head = factory.create(
        {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        }
    )
    assert [type(layer).__name__ for layer in head.layers] == ["Linear", "Dropout"]


# --------------------------------------------------------------------------- #
# Behavior: forward passes.
# --------------------------------------------------------------------------- #
def test_linear_forward(factory):
    enc = factory.create({"class_name": "linear", "in_features": 128, "out_features": 64})
    assert enc.forward(2.0) == 2.0 * 64 + 1.0  # bias on


def test_dropout_forward(factory):
    reg = factory.create({"class_name": "dropout", "p": 0.5})
    assert reg.forward(2.0) == 1.0


def test_sequential_forward_chains_children(factory):
    head = factory.create(
        {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        }
    )
    # linear (8x, no bias) -> 16.0 ; dropout (0.9x) -> 14.4
    assert head.forward(2.0) == pytest.approx(14.4)


# --------------------------------------------------------------------------- #
# Serialization round-trip.
# --------------------------------------------------------------------------- #
def test_to_json_round_trips_through_create(factory):
    enc = factory.create(
        {"class_name": "linear", "name": "enc", "in_features": 128, "out_features": 64}
    )
    clone = factory.create(enc.to_json())
    assert isinstance(clone, Linear)
    assert clone.config == enc.config


def test_to_json_preserves_discriminator(factory):
    enc = factory.create({"class_name": "linear", "in_features": 4, "out_features": 2})
    assert enc.to_json()["class_name"] == "linear"


# --------------------------------------------------------------------------- #
# Validation errors.
# --------------------------------------------------------------------------- #
def test_create_requires_mapping_with_class_name(factory):
    with pytest.raises(ValueError, match="class_name"):
        factory.create({"in_features": 4, "out_features": 2})
    with pytest.raises(ValueError, match="class_name"):
        factory.create("linear")  # not a mapping


def test_create_unknown_tag_raises(factory):
    with pytest.raises(UnknownTagError):
        factory.create({"class_name": "does-not-exist"})


def test_config_forbids_unknown_keys(factory):
    with pytest.raises(ValidationError):
        factory.create(
            {"class_name": "linear", "in_features": 4, "out_features": 2, "bogus": 1}
        )


def test_config_enforces_field_constraints(factory):
    with pytest.raises(ValidationError):
        factory.create({"class_name": "dropout", "p": 1.5})  # p must be <= 1.0


def test_config_requires_mandatory_fields(factory):
    with pytest.raises(ValidationError):
        factory.create({"class_name": "linear", "out_features": 2})  # missing in_features


def test_validate_returns_config_without_constructing(factory):
    cfg = factory.validate({"class_name": "linear", "in_features": 4, "out_features": 2})
    assert isinstance(cfg, Linear.Config)
    assert cfg.in_features == 4 and cfg.out_features == 2


# --------------------------------------------------------------------------- #
# Registry semantics.
# --------------------------------------------------------------------------- #
def test_registry_tags_preserve_insertion_order(factory):
    assert factory.registry.tags() == ["linear", "dropout", "sequential"]


def test_registry_rejects_conflicting_tag():
    reg = Registry()
    reg.register("x", Linear)
    with pytest.raises(DuplicateTagError):
        reg.register("x", Dropout)


def test_registry_re_register_same_class_is_idempotent():
    reg = Registry()
    reg.register("x", Linear)
    reg.register("x", Linear)  # no-op, must not raise
    assert reg.get("x") is Linear


def test_registry_unknown_tag_raises():
    reg = Registry()
    with pytest.raises(UnknownTagError):
        reg.get("missing")


# --------------------------------------------------------------------------- #
# Discriminated-union config adapter.
# --------------------------------------------------------------------------- #
def test_config_adapter_routes_by_discriminator(factory):
    adapter = factory.config_adapter()

    parsed = adapter.validate_python({"class_name": "dropout", "p": 0.25})
    assert isinstance(parsed, Dropout.Config)
    assert parsed.p == 0.25

    parsed = adapter.validate_python(
        {"class_name": "linear", "in_features": 2, "out_features": 3}
    )
    assert isinstance(parsed, Linear.Config)


def test_config_adapter_single_config_branch():
    f = ComponentFactory(Registry())
    f.register("dropout", Dropout)
    parsed = f.config_adapter().validate_python({"class_name": "dropout", "p": 0.25})
    assert isinstance(parsed, Dropout.Config)


def test_config_adapter_is_cached_until_registry_grows():
    f = ComponentFactory(Registry())
    f.register("dropout", Dropout)
    first = f.config_adapter()
    assert f.config_adapter() is first  # cached
    f.register("linear", Linear)
    assert f.config_adapter() is not first  # rebuilt after registry grew


def test_config_adapter_empty_registry_raises():
    with pytest.raises(ValueError, match="empty"):
        ComponentFactory(Registry()).config_adapter()
