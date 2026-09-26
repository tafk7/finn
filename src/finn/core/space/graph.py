# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Graph composition: link families, detached ends and topologies, interpretations.

A Space composes children as a graph. Nodes are its placements and structural
choices; ports are declared attachment points; nets are hyperedges placed as
``Link`` scopes. The engine owns structure, identity, evaluation and
attribution. A domain supplies meaning: its interfaces, link families, and
the interpretations that fold a topology into a product.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from ._configuration import Space
from .declarations import ViewKey, local_name
from .results import QueryResult
from .semantics import ValueSemantics, semantics_for

N = TypeVar("N")
E = TypeVar("E")
F = TypeVar("F")
R = TypeVar("R")


class Link(Space):
    """Base family of a net's scope.

    Declare ``Carried(interface)`` to read the unified carried value and
    ``Ends(interface)`` to read the active ends; both are computed by the
    engine from the net's attachments. Constraints, decisions and views
    declared here belong to each placed net.
    """


@dataclass(frozen=True, slots=True)
class End(Generic[F]):
    """One active end of a net, seen from inside the composite that owns the net.

    ``node`` names the composite's child (its declaration name), or is None for
    the composite's own port. ``direction`` is relative to the net: ``out``
    drives it and ``in`` reads it, so an input port of the composite is an
    ``out`` end inside. ``offer`` is the end's accepted facet, None for the
    composite's own port.
    """

    node: str | None
    port: str
    direction: Literal["in", "out"]
    publishes: bool
    offer: F | None = None


@dataclass(frozen=True, slots=True)
class EndRef:
    node: str | None
    port: str
    direction: Literal["in", "out"]


@dataclass(frozen=True, slots=True)
class NetEntry(Generic[E]):
    name: str
    value: E
    ends: tuple[EndRef, ...]


@dataclass(frozen=True, slots=True)
class PortEntry:
    name: str
    direction: Literal["in", "out"]


@dataclass(frozen=True, slots=True)
class Topology(Generic[N, E]):
    """The active topology of one composite, with each member's contribution.

    ``nodes`` and ``nets`` keep declaration order. A node or net whose family
    exports no contribution for the interpretation is omitted, as is every
    inactive node, net and port.
    """

    nodes: tuple[tuple[str, N], ...]
    nets: tuple[NetEntry[E], ...]
    ports: tuple[PortEntry, ...]


class Interpretation(Generic[N, E, R]):
    """A fold of a composite's topology into one product, defined once by a domain.

    ``node`` and ``net`` are the view keys through which children and link
    families contribute. ``reduce(topology=..., **arguments)`` is a pure
    function of the topology and the arguments bound by each ``Fold``.
    """

    def __init__(
        self,
        name: str,
        *,
        node: ViewKey[N] | None,
        net: ViewKey[E] | None,
        result: type[R] | ValueSemantics[R],
        reduce: Callable[..., R | QueryResult[R]],
    ) -> None:
        self.name = local_name(name, "interpretation name")
        self.node, self.net = node, net
        self.result: ValueSemantics[R] = semantics_for(result)
        self.reduce = reduce


TOPOLOGY: ValueSemantics[Topology[Any, Any]] = semantics_for(Topology)

__all__ = [
    "End",
    "EndRef",
    "Interpretation",
    "Link",
    "NetEntry",
    "PortEntry",
    "TOPOLOGY",
    "Topology",
]
