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
DependencyMode = Literal["required", "optional", "answer"]


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

    def __post_init__(self) -> None:
        for attr in ("members", "children", "named_members", "named_children"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))


@dataclass(frozen=True, slots=True)
class LinkedModel:
    nodes: tuple[Node, ...]
    scopes: tuple[Scope, ...]
    order: tuple[int, ...]
    parameters: tuple[int, ...]
    decisions: tuple[int, ...]
    keys: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
