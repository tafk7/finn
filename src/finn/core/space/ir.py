# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable records shared by compilation and evaluation.

A node (``Node``) is a record of a Space class while it is compiled (``_table``): its
references are the class's own records, by number, or edges out of the class
(``NodeRef``). A model reads each class's compiled form by address (``_scopes``).

A Space class's compiled form (``_template.Compiled``) holds its members (``Member``),
numbered within the class, their references edges (``Edge``): to a member of the class
(``Local``), to a member of a class it places, through its slots (``Placed``), to its own
formal (``Formal``), to the guard it is placed under (``Placing``), or to a selection of a
choice's member, in the choice's own scope (``Selected``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .semantics import ValueSemantics

if TYPE_CHECKING:
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
    "validity",
    "members",
]

# A node's reference while its class is compiled: a record of the class (its number), a
# member of a Space it places (``Placed``), a selection in a placed Space's choice
# (``Selected``), or the guard the class is placed under (``Placing``).
NodeRef: TypeAlias = "int | Placed | Selected | Placing"


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    node: NodeRef


@dataclass(frozen=True, slots=True)
class Node:
    index: int
    scope: int
    key: str
    kind: NodeKind
    semantics: ValueSemantics[object] | None = None
    guard: NodeRef | None = None
    arguments: tuple[Argument, ...] = ()
    function: Callable[..., object] | None = None
    call_style: Literal["explicit", "self"] = "explicit"
    value: object = None
    # A formal's: something must supply it.
    required: bool = True
    domain: Domain[object] | None = None
    domain_arguments: tuple[Argument, ...] = ()
    output: NodeRef | None = None
    # A validity node: its pieces, in order: the
    # Space's constraints, its Decisions (each a piece only where its domain lists no
    # value, or its guard is refused), and its placed Spaces' validity nodes.
    constraints: tuple[NodeRef, ...] = ()
    alternatives: tuple[tuple[str, NodeRef], ...] = ()
    selector: NodeRef | None = None
    source_owner: str | None = None
    origin: str | None = None

    @property
    def owner(self) -> str:
        """Authored owner for diagnostics from generated computation nodes."""
        return self.source_owner if self.source_owner is not None else self.key

    @property
    def dependencies(self) -> tuple[NodeRef, ...]:
        """Known structural/explicit edges; self-method reads are observed at runtime."""
        refs: list[NodeRef] = [arg.node for arg in (*self.arguments, *self.domain_arguments)]
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
