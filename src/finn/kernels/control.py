# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Control buses: a kernel presents its own, and its composition presents it at the top.

A kernel with a control interface (an AXI-Lite configuration bus, say) has a formal
naming the port its bus is presented at, given where it is placed (``control: str | None =
Param(default=None)``), and exports, under ``CONTROL``, the bus it presents there
with that port and the writes that put its configuration into the bus's registers
(``RegisterMap``): a ``Control``. A kernel's netlist view (its ``Members(CONTROL)``,
``finn.kernels.base.KernelSpace``) and a partition (``finn.kernels.partition``) each collect
their members' controls (``presented``: each a ``BusExport`` from its member's node) and
present each in their netlist (``finn.kernels.base.merged``), prefixing its port with the
member's node
(``first_s_axilite``); the top module renames the bus to that port
(``finn.kernels.artifacts.top.top_bus``), associated with its clock and reset. One kernel is
presented at one port.

A kernel declares its bus in its class body (``ControlBus``), after the members it
names::

    config = ControlBus(config_bus, presented=use_axilite, port=control, registers=register_map)
    config = ControlBus(config_bus)  # never presented: held idle

``bus`` is the kernel's ``Bus``; ``presented`` its bool member saying whether it presents
the bus (a choice of its own), or the constant False; ``port`` its formal naming the port
the bus is presented at; ``registers`` its ``RegisterMap``. It lowers to
``<name>_presented``, a derived value (bool), and ``<name>_control``, a view of its
``Control``, which the kernel exports as ``CONTROL``: ``Control(None)`` while it is not
presented. Its pins are the bus, and what it holds the bus's inputs while it is not
presented (``held_bus``: inputs constant, outputs unconnected): a declared bus is held or
presented by construction (``presents``: its inputs accounted for in its module), and
nothing about a bus is wired by name. Its pins come before every port's in its kernel's
module and its hold after theirs (``finn.kernels.declaration.ModuleOrder.ENCLOSING``). A kernel
declares one at most.

``<name>_presented`` is the kernel's own bus, presented: what a count of the control buses
sums, one a kernel. A boundary's AXI-Lite initiator is no kernel's bus and is not among
them (counted as a control bus, it would be counted again as a memory port).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from finn.core.space import (
    DefinitionError,
    Rejected,
    Space,
    View,
    ViewKey,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Bus, Direction, Pin
from finn.kernels.artifacts.module import BusExport, Held, RegisterMap
from finn.kernels.declaration import KernelDeclaration, ModuleOrder, Product, declared_reads

if TYPE_CHECKING:
    from finn.kernels.base import KernelSpace


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


@dataclass(frozen=True, eq=False)
class ControlBus(KernelDeclaration):
    """A kernel's control bus: its ``Bus``, whether it presents it, the port it is
    presented at and the writes of its configuration. The module docstring says what it
    lowers to."""

    bus: Any
    _: KW_ONLY
    presented: Any = False
    port: Any = None
    registers: Any = None

    KIND: ClassVar[str] = "control-bus"
    module_order: ClassVar[ModuleOrder] = ModuleOrder.ENCLOSING

    def lower(self, owner: type[KernelSpace], name: str) -> Mapping[str, object]:
        label = f"{owner.__qualname__}.{name}"
        self.references()  # each member it names, named
        if name == "bus":
            raise DefinitionError.broken(
                "declaration-member-taken",
                "ControlBus",
                f"{label} lowers to bus_control, which every kernel declares",
                declaration=label,
            )
        if self.presented is not False and (self.port is None or self.registers is None):
            raise DefinitionError.broken(
                "control-bus-unaddressed",
                "ControlBus",
                f"{label}: a bus it may present names the port it is presented at and its "
                "registers",
                declaration=label,
            )
        declaration = self

        @declared_reads(*self.paths("presented"))
        def presented_(self: Space) -> bool:
            """Whether it presents its control bus."""
            return bool(declaration.value(self, "presented"))

        @declared_reads(f"{name}_presented", *self.paths("bus", "port", "registers"))
        def control_(self: Space) -> Control | Rejected:
            """The control bus it presents, None while it presents none; a bus presented
            without the port it is presented at is refused."""
            if not getattr(self, f"{name}_presented"):
                return Control(None)
            port = declaration.value(self, "port")
            if port is None:
                return reject(
                    "control-bus-unaddressed", "a bus it presents names the port it is presented at"
                )
            return Control(
                declaration.value(self, "bus"), port, declaration.value(self, "registers")
            )

        def declares_(self: Space) -> bool:
            """It declares a control bus."""
            return True

        control = view(semantics=CONTROL_SEMANTICS)(control_)
        return {
            f"{name}_presented": derived(presented_),
            f"{name}_control": control,
            "declares_bus": derived(declares_),
            "bus_control": View(control),
        }

    def reads(self, name: str, product: Product) -> tuple[str, ...]:
        if product == "pins":
            return self.paths("bus")
        if product == "held":
            return (f"{name}_presented", *self.paths("bus"))
        return ()

    def pins(self, kernel: KernelSpace, name: str) -> tuple[Pin, ...] | Rejected:
        bus: Bus = self.value(kernel, "bus")
        return (bus,)

    def held(self, kernel: KernelSpace, name: str) -> Held | Rejected:
        if getattr(kernel, f"{name}_presented"):
            return Held()
        bus: Bus = self.value(kernel, "bus")
        return held_bus(bus)

    def presents(self, kernel: KernelSpace, name: str) -> bool:
        return bool(getattr(kernel, f"{name}_presented"))


__all__ = [
    "CONTROL",
    "Control",
    "ControlBus",
    "held_bus",
    "presented",
]
