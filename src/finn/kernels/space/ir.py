# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable linked representation shared by compilation and evaluation.

The table stores each occurrence once. Integer references describe potential
edges; the evaluator records demanded edges separately for every cached result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from .semantics import ValueSemantics

if TYPE_CHECKING:
    from .declarations import Space
    from .domains import Domain

NodeKind = Literal[
    "param",
    "const",
    "decision",
    "derived",
    "constraint",
    "view",
    "alias",
    "guard",
    "select",
    "readiness",
    "group",
]
DependencyMode = Literal["required", "optional", "result"]


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    node: int
    mode: DependencyMode = "required"


@dataclass(frozen=True, slots=True)
class Node:
    index: int
    scope: int
    key: str
    kind: NodeKind
    semantics: ValueSemantics[object] | None = None
    guard: int | None = None
    arguments: tuple[Argument, ...] = ()
    function: Callable[..., object] | None = None
    value: object = None
    required: bool = True
    domain: Domain[object] | None = None
    domain_arguments: tuple[Argument, ...] = ()
    output: int | None = None
    constraints: tuple[int, ...] = ()
    requires: tuple[int, ...] = ()
    alternatives: tuple[tuple[str, int], ...] = ()
    selector: int | None = None
    source_owner: str | None = None
    selection_index: Mapping[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Keep ordered alternatives as the structural graph. Only a genuine
        # multi-case selection needs a second, direct runtime lookup index.
        if self.kind == "select" and len(self.alternatives) > 1:
            object.__setattr__(self, "selection_index", MappingProxyType(dict(self.alternatives)))

    @property
    def owner(self) -> str:
        """Authored owner for diagnostics from generated computation nodes."""
        return self.source_owner if self.source_owner is not None else self.key

    @property
    def dependencies(self) -> tuple[int, ...]:
        """Conservative direct dependencies; never a transitive closure."""
        refs = [arg.node for arg in (*self.arguments, *self.domain_arguments)]
        if self.guard is not None:
            refs.append(self.guard)
        if self.output is not None:
            refs.append(self.output)
        if self.selector is not None:
            refs.append(self.selector)
        refs.extend(self.constraints)
        refs.extend(self.requires)
        refs.extend(target for _, target in self.alternatives)
        return tuple(dict.fromkeys(refs))


@dataclass(frozen=True, slots=True)
class Scope:
    index: int
    parent: int | None
    name: str
    space_type: type[Space]
    members: Mapping[object, int]
    children: Mapping[object, int] = field(default_factory=dict)
    named_members: Mapping[str, int] = field(default_factory=dict)
    named_children: Mapping[str, int] = field(default_factory=dict)
    guard: int | None = None
    choices: Mapping[object, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in ("members", "children", "named_members", "named_children", "choices"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))


@dataclass(frozen=True, slots=True)
class Choice:
    index: int
    scope: int
    key: str
    selector: int | None
    cases: tuple[tuple[str, int], ...]
    exports: Mapping[object, int]
    guard: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exports", MappingProxyType(dict(self.exports)))


@dataclass(frozen=True, slots=True)
class LinkedModel:
    nodes: tuple[Node, ...]
    scopes: tuple[Scope, ...]
    order: tuple[int, ...]
    parameters: tuple[int, ...]
    decisions: tuple[int, ...]
    keys: Mapping[str, int]
    choices: tuple[Choice, ...] = ()
    selector_choices: Mapping[int, int] = field(init=False, repr=False)
    ranks: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
        ranks = [0] * len(self.nodes)
        for rank, index in enumerate(self.order):
            ranks[index] = rank
        object.__setattr__(self, "ranks", tuple(ranks))
        object.__setattr__(
            self,
            "selector_choices",
            MappingProxyType(
                {item.selector: item.index for item in self.choices if item.selector is not None}
            ),
        )
