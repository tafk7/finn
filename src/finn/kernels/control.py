# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Control buses as ordinary Spaces that kernels reference.

A ``ControlBus`` is a node declared in the composite, named by the top-level
port it presents (``port``). A kernel with a
control interface (an AXI-Lite configuration bus, say) has a reference input
for it (``control: ControlBus = Param(required=False)``) and exports, under
``CONTROL``, the bus it presents there: ``exports = {CONTROL: {control:
control_bus}}``. The node sees its kernel through ``Users(CONTROL)`` and
exports an ``Exported`` bus under ``EXPORTED``; ``netlist`` renames the bus to
the node's port, associates it with the module's clock and reset, and wires it
through to the top. One kernel is controlled
through one bus node.

A kernel whose control interface is not referenced, or that exposes none in
its configuration (``Control(None)``), holds the bus inputs constant and
leaves its outputs unconnected through its ``Tieoffs``: nothing about a bus is
wired by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import (
    Param,
    Rejected,
    Space,
    Users,
    ViewKey,
    default_semantics,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Member
from finn.kernels.base import Tieoffs


@dataclass(frozen=True)
class Control:
    """The control bus a kernel presents on one input; None presents nothing."""

    bus: Bus | None


CONTROL_SEMANTICS = default_semantics(Control)
CONTROL = ViewKey("control", CONTROL_SEMANTICS)


@dataclass(frozen=True)
class Exported:
    """A kernel's control bus, the node it belongs to, and the top port it becomes."""

    node: str
    child: Bus
    port: str


EXPORTED_SEMANTICS = default_semantics(tuple)
EXPORTED = ViewKey("exported", EXPORTED_SEMANTICS)
"""A control bus node's exported bus: one ``Exported``, or none when unused."""


def top_bus(child: Bus, port: str, clock: str, reset: str) -> Bus:
    """The child's bus as a top target port: members renamed ``<port>_<MEMBER>``."""
    return Bus(
        port,
        child.protocol,
        tuple(
            Member(item.logical, f"{port}_{item.logical.upper()}", item.width)
            for item in child.signals
        ),
        Endpoint.TARGET,
        child.role,
        clock,
        reset,
    )


def held_bus(bus: Bus) -> Tieoffs:
    """A bus left unexposed: its inputs held low, its outputs unconnected."""
    directions = dict(bus.member_directions())
    inputs = tuple(
        (member.physical, 0)
        for member in bus.signals
        if directions[member.physical] is Direction.IN
    )
    unused = tuple(
        member.physical for member in bus.signals if directions[member.physical] is not Direction.IN
    )
    return Tieoffs(inputs, unused)


class ControlBus(Space):
    """A control interface of the composite: one kernel's bus, presented at ``port``."""

    port: str = Param()
    users = Users(CONTROL)

    @view(semantics=EXPORTED_SEMANTICS, requires=(users,))
    def exported(self) -> tuple[Exported, ...] | Rejected:
        present = [(str(user.node), user.value.bus) for user in self.users if user.value.bus]
        if len(present) > 1:
            named = ", ".join(node for node, _ in present)
            return reject("control-users", f"one kernel per control bus; referenced by {named}")
        return tuple(Exported(node, bus, self.port) for node, bus in present if bus is not None)

    exports = {EXPORTED: exported}


__all__ = [
    "CONTROL",
    "CONTROL_SEMANTICS",
    "Control",
    "ControlBus",
    "EXPORTED",
    "EXPORTED_SEMANTICS",
    "Exported",
    "held_bus",
    "top_bus",
]
