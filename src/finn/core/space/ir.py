# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable linked representation shared by compilation and evaluation.

The table stores each occurrence once. Node references (``NodeId``) describe
potential edges; the evaluator records demanded edges separately for every cached
result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

from .semantics import ValueSemantics

if TYPE_CHECKING:
    from ._configuration import Space
    from .domains import Domain

# A node's identity in its model: its position in ``LinkedModel.nodes``. Every
# reference to a node, here and in the evaluator, is a ``NodeId``, so a later
# identity (an instance, a template node) changes this name and the few places that
# turn a node into a cache key or a bit.
NodeId: TypeAlias = int

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
    "validity",
    "present",
    "members",
]


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    node: NodeId


@dataclass(frozen=True, slots=True)
class Node:
    index: NodeId
    scope: int
    key: str
    kind: NodeKind
    semantics: ValueSemantics[object] | None = None
    guard: NodeId | None = None
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
    output: NodeId | None = None
    # A view: its Space's validity node. A validity node: its pieces, in order: the
    # Space's constraints, its Decisions (each a piece only where its domain lists no
    # value, or its guard is refused), and its placed Spaces' validity nodes.
    constraints: tuple[NodeId, ...] = ()
    alternatives: tuple[tuple[str, NodeId], ...] = ()
    selector: NodeId | None = None
    source_owner: str | None = None
    origin: str | None = None
    # Structural edges that bypass forwarding aliases: (alias, its source).
    via: tuple[tuple[NodeId, NodeId], ...] = ()
    selection_index: Mapping[str, NodeId] | None = field(
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
    def dependencies(self) -> tuple[NodeId, ...]:
        """Known structural/explicit edges; self-method reads are observed at runtime."""
        refs = [arg.node for arg in (*self.arguments, *self.domain_arguments)]
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
    members: Mapping[object, NodeId]
    children: Mapping[object, int] = field(default_factory=dict)
    named_members: Mapping[str, NodeId] = field(default_factory=dict)
    named_children: Mapping[str, int] = field(default_factory=dict)
    guard: NodeId | None = None
    choices: Mapping[object, int] = field(default_factory=dict)
    # The node declaration instantiated here (None for a Space class compiled alone).
    record: object = None
    # The node answering this Space's validity (``kind == "validity"``).
    validity: NodeId | None = None

    def __post_init__(self) -> None:
        for attr in (
            "members",
            "children",
            "named_members",
            "named_children",
            "choices",
        ):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))


@dataclass(frozen=True, slots=True)
class Choice:
    """A Decision over nodes: its selector, candidate scopes (None places nothing),
    and the members read through it (``decision.member``) that were linked."""

    index: int
    scope: int
    key: str
    selector: NodeId
    cases: tuple[tuple[str, int | None], ...]
    members: Mapping[str, NodeId]
    guard: NodeId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", MappingProxyType(dict(self.members)))


@dataclass(frozen=True, slots=True)
class LinkedModel:
    nodes: tuple[Node, ...]
    scopes: tuple[Scope, ...]
    order: tuple[NodeId, ...]
    parameters: tuple[NodeId, ...]
    decisions: tuple[NodeId, ...]
    keys: Mapping[str, NodeId]
    choices: tuple[Choice, ...] = ()
    # Formals supplied by a named shared Decision: edits may pass through them.
    editable_aliases: frozenset[NodeId] = frozenset()
    # Each node's forwarding source: an alias chain collapsed to its source.
    forward: tuple[NodeId, ...] = ()
    # A selector's Choice (its index in ``choices``), by the selector node.
    selector_choices: Mapping[NodeId, int] = field(init=False, repr=False)
    ranks: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
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
