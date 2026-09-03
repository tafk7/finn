# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a source operand became in the selected Network, and nothing else.

The association answers one question: for this source tensor, where in the
selected Region graph does its data enter or leave, and how do its coordinates
correspond.  It is a *logical* record, and the exclusions are the contract:

```text
in      source node and operand, selected node/port or boundary,
        coordinate correspondence, source scope provenance

out     Kernel identity, component id, build unit, local physical-state
        destination, artifact key, memory layout, file path
```

A physical choice therefore cannot change an association.  Streaming a matrix
in through a boundary and producing it inside from a memstream are different
Networks and so associate differently -- but building the same Network with a
pumped core, an unpumped one, or a co-packaged one is the same association
every time, and U6 must not be able to quietly make that untrue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoordinateMapping(str, Enum):
    """How a source operand's index space corresponds to the selected one."""

    #: The same coordinates, in the same order.
    IDENTITY = "identity"
    #: Leading dimensions flattened into one, trailing dimension preserved.
    FLATTEN_LEADING = "flatten_leading"
    #: A two-dimensional operand read transposed.
    TRANSPOSE_2D = "transpose_2d"


@dataclass(frozen=True, slots=True)
class BoundaryDestination:
    """The operand crosses the Design's edge at a named boundary."""

    boundary: str
    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class StreamDestination:
    """The operand reaches a real port without crossing a boundary.

    A decoupled matrix produced inside the Design is still *traffic*: there is
    a port, it has a shape, and a consumer can look it up.
    """

    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class RegionStateDestination:
    """The operand is state of a node, and there is no port at all.

    An embedded matrix is baked into the node.  The distinction from
    :class:`StreamDestination` is the whole reason this is a union: reporting
    embedded state as a port named ``"embedded"`` names a port that does not
    exist, so a consumer resolving it finds nothing, and the empty shape that
    came with it reads as a zero-element tensor rather than as "no port".
    """

    node_id: str
    operand_id: str


#: Where one operand's data actually goes.  Three cases, because they are three
#: different facts and a caller acts differently on each.
OperandDestination = BoundaryDestination | StreamDestination | RegionStateDestination


@dataclass(frozen=True, slots=True)
class OperandAssociation:
    """One source operand and the selected place its data crosses."""

    operand: str
    tensor: str
    destination: OperandDestination
    correspondence: CoordinateMapping
    source_shape: tuple[int, ...]
    #: The shape at the destination port, or ``None`` when there is no port.
    #: Not ``()``: an empty tuple is the shape of a scalar.
    selected_shape: tuple[int, ...] | None

    @property
    def boundary(self) -> str | None:
        """The boundary id when this operand crosses the Design's edge."""

        return (
            self.destination.boundary if isinstance(self.destination, BoundaryDestination) else None
        )

    @property
    def node_id(self) -> str:
        return self.destination.node_id

    @property
    def port_id(self) -> str | None:
        """The port id, or ``None`` for an operand that is node state."""

        return (
            None
            if isinstance(self.destination, RegionStateDestination)
            else self.destination.port_id
        )


@dataclass(frozen=True, slots=True)
class SourceAssociation:
    """Every operand of one source node, against one selected Network.

    ``scope_id`` is the operation's own stable identity, not the node's display
    name.  A graph transformation that renames nodes must not silently
    reassociate anything, and a name is exactly what such a transformation is
    free to change.
    """

    scope_id: str
    source_node: str
    family: str
    family_version: str
    operands: tuple[OperandAssociation, ...]

    def operand(self, operand_id: str) -> OperandAssociation:
        for item in self.operands:
            if item.operand == operand_id:
                return item
        raise KeyError(operand_id)


__all__ = [
    "BoundaryDestination",
    "CoordinateMapping",
    "OperandAssociation",
    "OperandDestination",
    "RegionStateDestination",
    "SourceAssociation",
    "StreamDestination",
]
