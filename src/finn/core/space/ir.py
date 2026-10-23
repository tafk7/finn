# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable records shared by compilation and evaluation.

A node (``Node``) is a template's (``_template``), or one occurrence of a member in a
placed scope of a table (``LinkedModel``: ``_structure.expand``'s, read by the tests' line
oracles; ``Linked`` reads it), its references node ids (``NodeId``). A model reads neither:
it reads each class's compiled form by address (``_scopes``).

A Space class's compiled form (``_template.Compiled``) holds its members (``Member``),
numbered within the class, their references edges (``Edge``): to a member of the class
(``Local``), to a member of a class it places, through its slots (``Placed``), to its own
formal (``Formal``), to the guard it is placed under (``Placing``), or to a selection of a
choice's member, in the choice's own scope (``Selected``).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from .semantics import ValueSemantics

if TYPE_CHECKING:
    from ._configuration import Space
    from ._whole import Flow
    from .domains import Domain

# A node's identity in its model: in a model's structure (``_structure``), its scope's
# base plus its template node (a scope's block; numbered in the order scopes are
# addressed, so nothing orders by it); in a table, its position there.
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
    # A template's reference into a child Space: a node of the child class's template
    # (``_template``); never a node of a linked model.
    "import",
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
    # A formal's: something must supply it.
    required: bool = True
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


def forwards(node: Node | Member) -> bool:
    """A pure forwarding node: an alias that only passes its source's answer on."""
    return node.kind == "alias" and node.output is not None and node.domain is None


def guard_implies(nodes: Sequence[Node], required: int | None, guard: int | None) -> bool:
    """Whether ``required`` holds whenever ``guard`` does: it is ``guard`` or an outer guard."""
    if required is None:
        return True
    current = guard
    while current is not None:
        if current == required:
            return True
        current = nodes[current].guard
    return False


# -- a Space class's compiled form: its members, and their edges -------------------------------


@dataclass(frozen=True, slots=True)
class Local:
    """``(local, i)``: member ``i`` of the class itself."""

    member: int


@dataclass(frozen=True, slots=True)
class Placed:
    """``(slot k, j)``: member ``j`` of the class the slots ``path`` reach, from the class
    itself (a deeper path is a path of slots, each a slot of the class the one before
    reaches)."""

    path: tuple[int, ...]
    member: int


@dataclass(frozen=True, slots=True)
class Formal:
    """``(formal f)``: the class's own formal ``f``, read through the binding the placing
    scope writes for it."""

    name: str


@dataclass(frozen=True, slots=True)
class Placing:
    """The guard the class's scope is placed under: the placing scope's slot guard."""


PLACING = Placing()


@dataclass(frozen=True, slots=True)
class Selected:
    """``(choice, member name)`` in the scope the slots ``path`` reach (the class itself for
    no slots): the selection of the selected candidate's member, one address whoever reads
    it."""

    path: tuple[int, ...]
    choice: int
    member: str


# An edge of a member: to a member of the class, of a class it places, its own formal, the
# guard it is placed under, or a selection.
Edge: TypeAlias = "Local | Placed | Formal | Placing | Selected"


@dataclass(frozen=True, slots=True)
class Member:
    """A member of a Space class's compiled form, or a record the class writes for a Space it
    places (a formal's binding by reference, a selection it makes in that Space's choice): a
    node as its class states it, its references edges (``Edge``), numbered within the class
    (``index``)."""

    index: int
    key: str
    kind: NodeKind
    semantics: ValueSemantics[object] | None = None
    guard: Edge | None = None
    arguments: tuple[tuple[str, Edge], ...] = ()
    function: Callable[..., object] | None = None
    call_style: Literal["explicit", "self"] = "explicit"
    value: object = None
    required: bool = True
    domain: Domain[object] | None = None
    domain_arguments: tuple[tuple[str, Edge], ...] = ()
    output: Edge | None = None
    constraints: tuple[Edge, ...] = ()
    alternatives: tuple[tuple[str, Edge], ...] = ()
    selector: Edge | None = None
    source_owner: str | None = None
    origin: str | None = None

    @property
    def owner(self) -> str:
        """Authored owner for diagnostics from generated computation nodes."""
        return self.source_owner if self.source_owner is not None else self.key

    @property
    def dependencies(self) -> tuple[Edge, ...]:
        """Its edges, in ``Node.dependencies``' order."""
        refs: list[Edge] = [edge for _, edge in (*self.arguments, *self.domain_arguments)]
        if self.guard is not None:
            refs.append(self.guard)
        if self.output is not None:
            refs.append(self.output)
        if self.selector is not None:
            refs.append(self.selector)
        refs.extend(self.constraints)
        refs.extend(edge for _, edge in self.alternatives)
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
    # Its answers to the questions about the whole design space, where it is a structure's
    # (``_structure.materialize``), in its numbering; None for a table built by hand.
    whole: Questions | None = field(default=None, compare=False, repr=False)
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

    def header(self, index: NodeId) -> Node:
        """A node as a header reads it (``NodeHeader``): the node itself."""
        return self.nodes[index]

    def holds_scope(self, index: object) -> bool:
        """Whether ``index`` is one of its scopes: a table holds every scope, numbered from
        0."""
        return type(index) is int and 0 <= index < len(self.scopes)

    def holds_node(self, index: object) -> bool:
        """Whether ``index`` is one of its nodes: a table holds every node, numbered from
        0."""
        return type(index) is int and 0 <= index < len(self.nodes)


class Questions(Protocol):
    """The answers to the questions about a whole design space (``_whole``)."""

    def flow(self) -> Flow: ...

    def readers(self, source: NodeId) -> tuple[NodeId, ...]: ...


class NodeHeader(Protocol):
    """What is read of a node without resolving its references (but its guard's
    identity): a structure's ``Header``, or a table's ``Node``."""

    @property
    def index(self) -> NodeId: ...

    @property
    def scope(self) -> int: ...

    @property
    def key(self) -> str: ...

    @property
    def kind(self) -> NodeKind: ...

    @property
    def semantics(self) -> ValueSemantics[object] | None: ...

    @property
    def source_owner(self) -> str | None: ...

    @property
    def owner(self) -> str: ...

    @property
    def origin(self) -> str | None: ...

    @property
    def domain(self) -> Domain[object] | None: ...

    @property
    def guard(self) -> NodeId | None: ...


class _Ranked(Protocol):
    def __getitem__(self, index: int, /) -> Any: ...


class Linked(Protocol):
    """What the engine reads of a linked model: a model's structure (``_structure``), its
    nodes resolved on demand, or a table holding every node (``LinkedModel``: ``expand``'s,
    or one built by hand). ``ranks`` orders decisions (a structure's: by the flow order);
    ``header`` reads a node without resolving it; ``whole`` answers the questions about
    the whole design space (None for a table built by hand). ``holds_scope`` and
    ``holds_node`` say whether an identity is one of its scopes or nodes, the same way in
    every mode (never by comparing it with a length: a structure's ``NodeId``s come in
    blocks with holes). An identity standing for another node is not one of its nodes:
    ``holds_node`` is the question for an identity, not for a read, and never follows it
    (a table has none)."""

    @property
    def nodes(self) -> Sequence[Node]: ...

    @property
    def scopes(self) -> Sequence[Scope]: ...

    @property
    def choices(self) -> Sequence[Choice]: ...

    @property
    def keys(self) -> Mapping[str, NodeId]: ...

    @property
    def selector_choices(self) -> Mapping[NodeId, int]: ...

    @property
    def editable_aliases(self) -> Collection[NodeId]: ...

    @property
    def forward(self) -> Sequence[NodeId]: ...

    @property
    def parameters(self) -> tuple[NodeId, ...]: ...

    @property
    def decisions(self) -> tuple[NodeId, ...]: ...

    @property
    def ranks(self) -> _Ranked: ...

    @property
    def whole(self) -> Questions | None: ...

    def header(self, index: NodeId, /) -> NodeHeader: ...

    def holds_scope(self, index: object, /) -> bool: ...

    def holds_node(self, index: object, /) -> bool: ...
