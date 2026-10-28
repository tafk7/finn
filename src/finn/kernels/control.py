# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Included control-bus model inputs and pure physical assembly.

A kernel includes ControlModel(bus=config_bus, presented=use_axilite). Its
parent bus_control View reads config.presented first, then its control port,
config.bus and register_map View; the false path returns Control(None) and
skips port/register inputs. RegisterMap never becomes a Group Param.

Public physical Views share the enclosing kernel's ownership. Module builders
call pin_values/held_values directly, retaining control-before-ports pins and
ports-before-control held order. The kernel declares its own bus_control and
declares_bus answers. control_models discovers only lexical Group occurrences,
not controls of child kernels. Every kernel declares at most one control bus.

presented collects actual Control values as BusExports in their existing order;
held_bus retains input/output physical directions without numerical changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finn.core.space import Group, Param, Rejected, Space, reject, view
from finn.kernels.artifacts.abi import Bus, Direction, Pin
from finn.kernels.artifacts.module import BusExport, Held, RegisterMap


@dataclass(frozen=True)
class Control:
    """The control bus a kernel presents, None presenting nothing; the port its placement
    names for it; and the writes that put its configuration into the bus's registers."""

    bus: Bus | None
    port: str = ""
    registers: RegisterMap = RegisterMap()


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


class ControlModel(Group):
    """Included model/physical inputs; parent owns its port and register-map View.

    Group ownership is the enclosing real Space. Serial module assembly calls
    the ordinary methods; public physical Views keep that owning validity.
    """

    bus: Bus = Param()
    presented: bool = Param(default=False)

    def pin_values(self) -> tuple[Pin, ...]:
        return (self.bus,)

    def held_values(self) -> Held:
        if self.presented:
            return Held()
        return held_bus(self.bus)

    @view
    def physical_pins(self) -> tuple[Pin, ...]:
        return self.pin_values()

    @view
    def idle_pins(self) -> Held:
        return self.held_values()


def control_models(space_type: type[Space]) -> dict[str, ControlModel]:
    """Effective included controls through lexical Groups, stopping at real Spaces.

    This is structural authoring discovery, not a model binding or read list.
    References on Group occurrences retain their actual enclosing placement.
    """

    def below(
        namespace: type[Space] | type[Group], placed: Group | None, prefix: str
    ) -> dict[str, ControlModel]:
        effective: dict[str, object] = {}
        for base in reversed(namespace.__mro__):
            effective.update(vars(base))
        found: dict[str, ControlModel] = {}
        for name, declaration in effective.items():
            if isinstance(declaration, ControlModel):
                control = declaration if placed is None else getattr(placed, name)
                found[prefix + name] = control
            elif isinstance(declaration, Group):
                group = declaration if placed is None else getattr(placed, name)
                found.update(below(type(declaration), group, prefix + name + "."))
        return found

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise TypeError("control_models requires a real Space class")
    return below(space_type, None, "")


__all__ = [
    "Control",
    "ControlModel",
    "control_models",
    "held_bus",
    "presented",
]
