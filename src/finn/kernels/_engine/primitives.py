# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small, domain-free primitives shared by the engine."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Generic, NoReturn, Protocol, TypeAlias, TypeVar, cast

T = TypeVar("T")

_PATH_RE = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", re.ASCII)


@total_ordering
@dataclass(frozen=True, slots=True)
class QualifiedPath:
    """An ASCII, byte-ordered declaration or problem-field path."""

    value: str

    def __post_init__(self) -> None:
        if _PATH_RE.fullmatch(self.value) is None:
            raise ValueError(
                "qualified paths contain one or more non-empty ASCII segments using only "
                "letters, digits, '_' and '-'"
            )

    @classmethod
    def parse(cls, value: QualifiedPath | str) -> QualifiedPath:
        return value if isinstance(value, QualifiedPath) else cls(value)

    def __str__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, QualifiedPath):
            return NotImplemented
        return self.value.encode("ascii") < other.value.encode("ascii")


PathMapping: TypeAlias = (
    Mapping[QualifiedPath, object] | Mapping[str, object] | Mapping[QualifiedPath | str, object]
)


class NoTruthValue:
    """Mixin for scalar semantic answers that must be inspected explicitly."""

    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} has no truth value; inspect its fields")


@dataclass(frozen=True, slots=True)
class ValueSemantics(Generic[T]):
    """Adapter-owned runtime recognition, equality, and snapshot policy."""

    type_token: object
    name: str
    recognizes: Callable[[object], bool]
    equal: Callable[[T, T], bool]
    snapshot: Callable[[T], T]

    @classmethod
    def immutable_nominal(
        cls, value_type: type[T], *, name: str | None = None
    ) -> ValueSemantics[T]:
        return cls(
            type_token=value_type,
            name=name or value_type.__qualname__,
            recognizes=lambda value: type(value) is value_type,
            equal=lambda left, right: left == right,
            snapshot=lambda value: value,
        )

    def accepts(self, value: object) -> bool:
        return bool(self.recognizes(value))

    def freeze(self, value: object) -> T:
        if not self.accepts(value):
            raise TypeError(f"expected value of nominal type {self.name}")
        frozen = self.snapshot(cast(T, value))
        if not self.accepts(frozen):
            raise TypeError(f"snapshot for {self.name} changed its nominal value type")
        return frozen

    def values_equal(self, left: object, right: object) -> bool:
        if not self.accepts(left) or not self.accepts(right):
            return False
        return bool(self.equal(cast(T, left), cast(T, right)))

    def is_compatible_with(self, other: ValueSemantics[object]) -> bool:
        return self.type_token is other.type_token


class _Sortable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


K = TypeVar("K", bound=_Sortable)


def stable_topological_order(graph: Mapping[K, AbstractSet[K]]) -> tuple[K, ...]:
    """Order dependencies first, appending cyclic nodes in stable order."""

    pending = {node: set(dependencies) for node, dependencies in graph.items()}
    dependents: dict[K, list[K]] = {}
    for node, dependencies in pending.items():
        for dependency in dependencies:
            dependents.setdefault(dependency, []).append(node)
    order: list[K] = []
    layer = sorted(node for node, dependencies in pending.items() if not dependencies)
    while layer:
        order.extend(layer)
        following: list[K] = []
        for node in layer:
            del pending[node]
            for dependent in dependents.get(node, ()):
                remaining = pending.get(dependent)
                if remaining is None:
                    continue
                remaining.discard(node)
                if not remaining:
                    following.append(dependent)
        layer = sorted(following)
    order.extend(sorted(pending))
    return tuple(order)


__all__ = [
    "NoTruthValue",
    "PathMapping",
    "QualifiedPath",
    "ValueSemantics",
    "stable_topological_order",
]
