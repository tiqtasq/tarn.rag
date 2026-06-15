"""
Class registry for components. The class registry owns the tag -> class mapping (a pure lookup table).
"""

from __future__ import annotations

from collections.abc import Iterator


class DuplicateTagError(ValueError):
    """Two classes tried to register under the same tag."""


class UnknownTagError(KeyError):
    """A spec referenced a tag that is not registered."""


# --------------------------------------------------------------------------- #
# Registry: the tag -> class mapping. No construction logic lives here.       #
# --------------------------------------------------------------------------- #
class Registry[T]:

    def __init__(self) -> None:
        self._map: dict[str, type[T]] = {}

    def register(self, tag: str, cls: type[T]) -> None:
        existing = self._map.get(tag)
        if existing is not None and existing is not cls:
            raise DuplicateTagError(
                f"tag {tag!r} is already registered to {existing.__qualname__}"
            )
        self._map[tag] = cls

    def get(self, tag: str) -> type[T]:
        try:
            return self._map[tag]
        except KeyError:
            raise UnknownTagError(
                f"unknown tag {tag!r}; registered: {sorted(self._map)}"
            ) from None

    def __contains__(self, tag: object) -> bool:
        return tag in self._map

    def __iter__(self) -> Iterator[str]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def tags(self) -> list[str]:
        return list(self._map)

    def classes(self) -> list[type[T]]:
        return list(self._map.values())

    def items(self) -> list[tuple[str, type[T]]]:
        return list(self._map.items())

    def clear(self) -> None:
        self._map.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._map)!r})"
