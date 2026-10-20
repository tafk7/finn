# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Control buses: a kernel presents its own, and its composition presents it at the top.

A kernel with a control interface (an AXI-Lite configuration bus, say) has a formal
naming the port its bus is presented at, given where it is placed (``control: str =
Param(required=False)``), and exports, under ``CONTROL``, the bus it presents there
with that port and the writes that put its configuration into the bus's registers
(``RegisterMap``): a ``Control``. A kernel with children (``Members(CONTROL)``) and a
network (``finn.kernels.network``) each collect their members' controls
(``presented``: each a ``BusExport`` from its member's node) and present each in their
netlist (``finn.kernels.base.merged``), prefixing its port with the member's node
(``first_s_axilite``); the top module renames the bus to that port
(``finn.kernels.artifacts.top.top_bus``), associated with its clock and reset. One kernel is
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
from finn.kernels.artifacts.abi import Bus, Direction
from finn.kernels.artifacts.module import BusExport, Held, RegisterMap


@dataclass(frozen=True)
class Control:
    """The control bus a kernel presents, None presenting nothing; the port its placement
    names for it; and the writes that put its configuration into the bus's registers."""

    bus: Bus | None
    port: str = ""
    registers: RegisterMap = RegisterMap()


CONTROL_SEMANTICS = default_semantics(Control)
CONTROL = ViewKey("control", CONTROL_SEMANTICS)


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


def presented(controls: Iterable[tuple[str, Control]]) -> tuple[BusExport, ...] | Rejected:
    """The buses ``controls`` present, each a member's node and its ``Control``, in order:
    each exported from its node at its port, with the writes its configuration takes; a
    port two members present at is refused (``control-users``)."""
    found = tuple(
        (node, control.bus, control) for node, control in controls if control.bus is not None
    )
    by_port: dict[str, list[str]] = {}
    for node, _, control in found:
        by_port.setdefault(control.port, []).append(node)
    shared = {port: nodes for port, nodes in by_port.items() if len(nodes) > 1}
    if shared:
        named = "; ".join(f"{port} by {', '.join(nodes)}" for port, nodes in shared.items())
        return reject("control-users", f"one kernel per control bus; presented at {named}")
    return tuple(
        BusExport(node, bus, control.port, control.registers) for node, bus, control in found
    )


__all__ = [
    "CONTROL",
    "Control",
    "held_bus",
    "presented",
]
