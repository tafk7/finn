# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Control buses: a kernel presents its own, and its composition presents it at the top.

A kernel with a control interface (an AXI-Lite configuration bus, say) has a formal
naming the port its bus is presented at, given where it is placed (``control: str =
Param(required=False)``), and exports, under ``CONTROL``, the bus it presents there
with that port and the writes that put its configuration into the bus's registers
(``RegisterMap``): a ``Control``. A kernel with children (``Members(CONTROL)``) and a
network (``finn.kernels.network``) each collect their members' controls
(``presented``) and present each in their netlist (``BusExport``), prefixing its port
with the member's node (``first_s_axilite``); the top module renames the bus to that port
(``top_bus``), associated with its clock and reset. One kernel is
presented at one port.

A kernel not given a port, or that exposes no bus in its configuration
(``Control(None)``), holds the bus inputs constant and leaves its outputs unconnected
(``held_bus``), and presents it otherwise (``LeafKernel.controlled``): nothing about a
bus is wired by name.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finn.core.space import Rejected, ViewKey, default_semantics, reject
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Member
from finn.kernels.artifacts.module import Held, RegisterMap


@dataclass(frozen=True)
class Control:
    """The control bus a kernel presents, None presenting nothing; the port its placement
    names for it; and the writes that put its configuration into the bus's registers."""

    bus: Bus | None
    port: str = ""
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


def presented(controls: Iterable[tuple[str, Control]]) -> tuple[Exported, ...] | Rejected:
    """The buses ``controls`` present, each a member's node and its ``Control``, in order;
    a port two members present at is refused (``control-users``)."""
    found = tuple(
        Exported(node, control.bus, control.port, control.registers)
        for node, control in controls
        if control.bus is not None
    )
    by_port: dict[str, list[str]] = {}
    for item in found:
        by_port.setdefault(item.port, []).append(item.node)
    shared = {port: nodes for port, nodes in by_port.items() if len(nodes) > 1}
    if shared:
        named = "; ".join(f"{port} by {', '.join(nodes)}" for port, nodes in shared.items())
        return reject("control-users", f"one kernel per control bus; presented at {named}")
    return found


__all__ = [
    "CONTROL",
    "Control",
    "Exported",
    "held_bus",
    "presented",
    "top_bus",
]
