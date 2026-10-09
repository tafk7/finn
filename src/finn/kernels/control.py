# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Control buses as ordinary Spaces that kernels reference.

A ``ControlBus`` is a node declared in a kernel with children, named by the
port it presents (``port``). A kernel with a control interface (an AXI-Lite
configuration bus, say) has a reference input for it (``control: ControlBus =
Param(required=False)``) and exports, under ``CONTROL``, the bus it presents
there: ``exports = {CONTROL: {control: control_bus}}``. The node sees its
kernel through ``Users(CONTROL)`` and exports an ``Exported`` bus under
``EXPORTED``; the kernel that declares the node presents it in its netlist
(``BusExport``), each parent prefixing its port with the child's node
(``first_s_axilite``), and the root's module renames the bus to that port
(``top_bus``), associated with its clock and reset. The writes that put the
kernel's configuration into the bus's registers (``RegisterMap``) travel with it
to the root's ``BusExport``. One kernel is controlled through one bus node.

A kernel whose control interface is not referenced, or that exposes none in
its configuration (``Control(None)``), holds the bus inputs constant and
leaves its outputs unconnected (``held_bus``), and presents it otherwise
(``Kernel.controlled``): nothing about a bus is wired by name.
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
from finn.kernels.artifacts.module import Held, RegisterMap


@dataclass(frozen=True)
class Control:
    """The control bus a kernel presents on one input, None presenting nothing, and the
    writes that put its configuration into the bus's registers."""

    bus: Bus | None
    registers: RegisterMap = RegisterMap()


CONTROL_SEMANTICS = default_semantics(Control)
CONTROL = ViewKey("control", CONTROL_SEMANTICS)


@dataclass(frozen=True)
class Exported:
    """A kernel's control bus, the node it belongs to, the top port it becomes, and the
    writes its configuration takes."""

    node: str
    child: Bus
    port: str
    registers: RegisterMap = RegisterMap()


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


def held_bus(bus: Bus) -> Held:
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
    return Held(inputs, unused)


class ControlBus(Space):
    """A control interface of a kernel with children: one kernel's bus, presented at ``port``."""

    port: str = Param()
    users = Users(CONTROL)

    @view(requires=(users,))
    def exported(self) -> tuple[Exported, ...] | Rejected:
        present = [(str(user.node), user.value) for user in self.users if user.value.bus]
        if len(present) > 1:
            named = ", ".join(node for node, _ in present)
            return reject("control-users", f"one kernel per control bus; referenced by {named}")
        return tuple(
            Exported(node, control.bus, self.port, control.registers)
            for node, control in present
            if control.bus is not None
        )

    exports = {EXPORTED: exported}


__all__ = [
    "CONTROL",
    "Control",
    "ControlBus",
    "EXPORTED",
    "Exported",
    "held_bus",
    "top_bus",
]
