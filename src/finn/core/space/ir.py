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
    from ._configuration import Space
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
    "group",
    "present",
    "locate",
    "members",
]


@dataclass(frozen=True, slots=True)
class Layer:
    """One body's setting of a member: who wrote it, where, and what."""

    body: str
    origin: str | None
    value: str
    # The Space class's own declaration (a Param default, a Decision, a child node).
    declared: bool = False

    def describe(self) -> str:
        where = f" at {self.origin}" if self.origin else ""
        if self.declared:
            return f"declared {self.value}{where}"
        return f"{self.value} set by {self.body}{where}"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Who set a member's effective value, and what it overrides.

    ``layers`` runs from the Space class's declaration outwards; the last layer is
    the effective one (the outermost body that set it wins).
    """

    key: str
    layers: tuple[Layer, ...]

    @property
    def effective(self) -> Layer:
        return self.layers[-1]

    @property
    def overridden(self) -> tuple[Layer, ...]:
        return self.layers[:-1]

    def text(self) -> str:
        """``kitchen.area = 16 (set by House at house.py:42; declared 12 at room.py:10)``."""
        effective = self.effective
        where = f" at {effective.origin}" if effective.origin else ""
        head = "declared" if effective.declared else f"set by {effective.body}"
        parts = [f"{head}{where}"]
        parts.extend(
            ("overrides " if not layer.declared else "") + layer.describe()
            for layer in reversed(self.overridden)
        )
        return f"{self.key} = {effective.value} ({'; '.join(parts)})"

    def __str__(self) -> str:
        return self.text()


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    node: int


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
    call_style: Literal["explicit", "self"] = "explicit"
    value: object = None
    required: bool = True
    # A Decision with no safe baseline (``Decision(required=True)``); ``required``
    # above is a formal's, that something must supply it.
    required_choice: bool = False
    domain: Domain[object] | None = None
    domain_arguments: tuple[Argument, ...] = ()
    output: int | None = None
    constraints: tuple[int, ...] = ()
    alternatives: tuple[tuple[str, int], ...] = ()
    selector: int | None = None
    source_owner: str | None = None
    origin: str | None = None
    # A Decision replaced by a narrower one keeps its declared domain as a contract;
    # a pinned coordinate (a const or alias with a domain) is checked against it.
    contract: Domain[object] | None = None
    contract_arguments: tuple[Argument, ...] = ()
    # Provenance text for refusals of a supplied value.
    note: str | None = None
    # Structural edges that bypass forwarding aliases: (alias, its source).
    via: tuple[tuple[int, int], ...] = ()
    selection_index: Mapping[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Keep ordered alternatives as the structural graph; a selection also
        # needs a direct runtime lookup (a case without the member is absent).
        if self.kind == "select":
            object.__setattr__(self, "selection_index", MappingProxyType(dict(self.alternatives)))

    @property
    def owner(self) -> str:
        """Authored owner for diagnostics from generated computation nodes."""
        return self.source_owner if self.source_owner is not None else self.key

    @property
    def dependencies(self) -> tuple[int, ...]:
        """Known structural/explicit edges; self-method reads are observed at runtime."""
        refs = [
            arg.node for arg in (*self.arguments, *self.domain_arguments, *self.contract_arguments)
        ]
        if self.guard is not None:
            refs.append(self.guard)
        if self.output is not None:
            refs.append(self.output)
        if self.selector is not None:
            refs.append(self.selector)
        refs.extend(self.constraints)
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
    # The node declaration instantiated here (None for a Space class compiled alone).
    record: object = None
    # Reference inputs: the formal's declaration -> the scope of the node it
    # references, which is placed elsewhere (not a child of this scope).
    references: Mapping[object, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in (
            "members",
            "children",
            "named_members",
            "named_children",
            "choices",
            "references",
        ):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))


@dataclass(frozen=True, slots=True)
class Choice:
    """A Decision over nodes: its selector, candidate scopes (None places nothing),
    and the members read through it (``decision.member``) that were linked."""

    index: int
    scope: int
    key: str
    selector: int
    cases: tuple[tuple[str, int | None], ...]
    members: Mapping[str, int]
    guard: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", MappingProxyType(dict(self.members)))


@dataclass(frozen=True, slots=True)
class LinkedModel:
    nodes: tuple[Node, ...]
    scopes: tuple[Scope, ...]
    order: tuple[int, ...]
    parameters: tuple[int, ...]
    decisions: tuple[int, ...]
    keys: Mapping[str, int]
    choices: tuple[Choice, ...] = ()
    # Formals supplied by a named shared Decision: edits may pass through them.
    editable_aliases: frozenset[int] = frozenset()
    # Who set each supplied member (by node) and each replaced child (by scope).
    provenance: Mapping[int, Provenance] = field(default_factory=dict)
    scope_provenance: Mapping[int, Provenance] = field(default_factory=dict)
    # Decision keys removed by an override that pinned the coordinate.
    pinned: Mapping[str, Provenance] = field(default_factory=dict)
    # Each node's forwarding source: an alias chain collapsed to its source.
    forward: tuple[int, ...] = ()
    selector_choices: Mapping[int, int] = field(init=False, repr=False)
    ranks: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
        for attr in ("provenance", "scope_provenance", "pinned"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))
        if not self.forward:
            object.__setattr__(self, "forward", tuple(range(len(self.nodes))))
        ranks = [0] * len(self.nodes)
        for rank, index in enumerate(self.order):
            ranks[index] = rank
        object.__setattr__(self, "ranks", tuple(ranks))
        object.__setattr__(
            self,
            "selector_choices",
            MappingProxyType({item.selector: item.index for item in self.choices}),
        )
