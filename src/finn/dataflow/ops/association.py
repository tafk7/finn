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
class OperandAssociation:
    """One source operand and the selected place its data crosses."""

    operand: str
    tensor: str
    #: The selected Network *boundary* id when the data crosses the Design's
    #: edge, and ``None`` when it does not -- an embedded matrix crosses no
    #: boundary, which is a fact about the Network and is recorded as one.
    boundary: str | None
    #: The Region node and port the data reaches, whether or not it crossed a
    #: boundary to get there.
    node_id: str
    port_id: str
    correspondence: CoordinateMapping
    source_shape: tuple[int, ...]
    selected_shape: tuple[int, ...]


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
    "CoordinateMapping",
    "OperandAssociation",
    "SourceAssociation",
]
