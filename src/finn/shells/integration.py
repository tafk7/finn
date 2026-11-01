# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The integration description: what a shell's linker builds around a build's cores, with no
platform's names in it, and what a linker gives back.

``describe(model)`` takes the build's parent graph (its one partition node,
``finn.core.graph.partition.partition_body``), refuses a body whose design its check did not
verify as it is now (``refuse_unverified``), configures the partition's network as hardware
generation does (``configured_network``), and reads the movers its shell places
(``finn.kernels.shell``, each its contract, ``IodmaKernel.contract``, and its module) and its
shell's row. It states (``Integration``):

- the row it was made for, its linker, its runtime (the row's first), the part and the
  period asked;
- each core (``Core``): its node's name, and per edge port its supply (``Supply``) or
  delivery (``Delivery``): the tensor, its shape and element, the offer's kind, the mover's
  unit, the memory target it binds, the mover's facts (``MoverContract``) and its module,
  whose IP its unit is;
- the units the platform places (``Unit``): each core's, named as its node, and each
  mover's, ``in<i>`` and ``out<j>``, its name in the shell (by its direction and its edge
  port's index), each with its kind, its IP (``IpRef``) and its control window; each
  named once (a core named as a mover's unit is refused, ``unit-name-collides``);
- the connections (``Connection``): each mover's stream to its core's edge port, with its
  bits;
- the bindings (``Binding``): each mover's memory port to a memory target of the row;
- the endpoints (``Endpoint``): each control bus, by unit and bus, at its offset in its unit
  and its range;
- the clocks (``ClockPin``): each unit's clock and reset pins, on a clock the row names.

Every port is its unit's own pin: a core's edge ports and buses its module's, a mover's
stream, memory port and control bus its module's (``mover_pins``). Where each unit is
placed and what its cell is called are the linker's (``Linker``):
``finn.shells.pynq.linker`` for the ``pynq`` shell. A mover's IP is its module, built by
the shell's runner (its unit's ``ip`` is ``None`` until then).

Each mover binds the row's first memory target. Until the core unit has a control
grouper, a core's unit has no window of its own (``Unit.window`` is ``None``) and each of
its kernels' control buses is an endpoint at offset 0 with its own range: ``2**awaddr``
bytes, and the row's least window at least; a mover's control bus is one, of its own
range, in its unit's window.

It is a function of a configured point: the description is a build's, needed once a build
and not at every point an exploration costs. A shell without a linker (``ip``) has no
description: its user links the core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from finn.core.graph.annotations.platform import read_target
from finn.core.graph.facts import known_shape
from finn.core.graph.partition import partition_body
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import Available
from finn.custom_op.kernels.shell import configured_network, mover_contracts, refuse_unverified
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.artifacts.abi import Bus, StandardProtocol
from finn.kernels.artifacts.ipxact import vlnv
from finn.kernels.artifacts.module import Leaf, module_name
from finn.kernels.library import library_module
from finn.kernels.movers import MoverContract
from finn.platform.shells import placed_offer
from finn.transformation.kernels.package import source_roots

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.toolchain.launcher import Toolchain

CORE = "core"
"""A core unit's kind."""

MAIN = "main"
"""The row's main clock (``finn.platform.shells.Clocks``)."""

CLOCK, RESET = "ap_clk", "ap_rst_n"
"""Every unit's clock and reset pins."""


@dataclass(frozen=True)
class IpRef:
    """A unit's IP: its ``vlnv`` and the IP ``repositories`` a tool adds to find it, none
    until it is packaged."""

    vlnv: str
    repositories: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Unit:
    """What the platform places: its ``name``, its ``kind`` (``core``, or a mover's kind),
    its ``ip`` (``None`` until the shell's runner builds it) and the bytes of its control
    ``window`` (``None``: none of its own)."""

    name: str
    kind: str
    ip: IpRef | None
    window: int | None


@dataclass(frozen=True, kw_only=True)
class Edge:
    """An edge port of a core and its mover: the core's ``port``, the boundary ``tensor``,
    its ``shape`` (one inference, as the graph states it) and ``element``, the offer's
    ``kind``, the mover's ``unit``, the memory ``target`` its memory port binds, the
    mover's facts (``contract``) and its ``module``, whose IP its unit is."""

    port: str
    tensor: str
    shape: tuple[int, ...]
    element: ScalarEncoding
    kind: str
    unit: str
    target: str
    contract: MoverContract
    module: Leaf


@dataclass(frozen=True, kw_only=True)
class Supply(Edge):
    """An edge input port of a core and what supplies it."""


@dataclass(frozen=True, kw_only=True)
class Delivery(Edge):
    """An edge output port of a core and what delivers it."""


@dataclass(frozen=True, kw_only=True)
class Core:
    """A core: its ``name`` (its node's, and its unit's), and its edge's ``inputs`` and
    ``outputs`` in port order."""

    name: str
    inputs: tuple[Supply, ...]
    outputs: tuple[Delivery, ...]


@dataclass(frozen=True)
class Connection:
    """A stream from ``source`` to ``sink``, each ``<unit>/<port>``, of ``bits`` a beat."""

    source: str
    sink: str
    bits: int


@dataclass(frozen=True)
class Binding:
    """A memory port (``<unit>/<port>``) bound to a memory ``target`` the row names."""

    port: str
    target: str


@dataclass(frozen=True, kw_only=True)
class Endpoint:
    """A control bus: its ``unit``, the ``bus`` inside it, at ``offset`` in the unit's window,
    of ``range`` bytes."""

    unit: str
    bus: str
    offset: int
    range: int


@dataclass(frozen=True)
class ClockPin:
    """A unit's ``pin`` and its ``reset`` pin, on the row's ``clock``."""

    unit: str
    pin: str
    reset: str
    clock: str


@dataclass(frozen=True, kw_only=True)
class Integration:
    """The integration description of a build's cores in their shell (see the module
    docstring)."""

    row: tuple[str, str | None]
    linker: str
    runtime: str | None
    part: str
    period_ns: float
    cores: tuple[Core, ...]
    units: tuple[Unit, ...]
    connections: tuple[Connection, ...]
    bindings: tuple[Binding, ...]
    endpoints: tuple[Endpoint, ...]
    clocks: tuple[ClockPin, ...]

    def unit(self, name: str) -> Unit:
        """The unit named ``name``."""
        (found,) = (unit for unit in self.units if unit.name == name)
        return found

    def packaged(self, ips: Mapping[str, IpRef]) -> Integration:
        """This description, each unit named in ``ips`` given that IP."""
        units = tuple(replace(unit, ip=ips.get(unit.name, unit.ip)) for unit in self.units)
        return replace(self, units=units)

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Every core's supplies and deliveries, the cores in order, inputs first."""
        return tuple(edge for core in self.cores for edge in (*core.inputs, *core.outputs))

    def report(self) -> dict[str, Any]:
        """The description as JSON."""

        def edge(each: Edge) -> dict[str, Any]:
            return {
                "port": each.port,
                "tensor": each.tensor,
                "shape": list(each.shape),
                "element": each.element.dtype.name,
                "kind": each.kind,
                "unit": each.unit,
                "target": each.target,
                "mover": each.contract.kind,
            }

        return {
            "row": {"shell": self.row[0], "board": self.row[1]},
            "linker": self.linker,
            "runtime": self.runtime,
            "part": self.part,
            "period_ns": self.period_ns,
            "cores": [
                {
                    "name": core.name,
                    "inputs": [edge(each) for each in core.inputs],
                    "outputs": [edge(each) for each in core.outputs],
                }
                for core in self.cores
            ],
            "units": [
                {
                    "name": unit.name,
                    "kind": unit.kind,
                    "ip": None
                    if unit.ip is None
                    else {"vlnv": unit.ip.vlnv, "repositories": list(unit.ip.repositories)},
                    "window": unit.window,
                }
                for unit in self.units
            ],
            "connections": [
                {"source": each.source, "sink": each.sink, "bits": each.bits}
                for each in self.connections
            ],
            "bindings": [{"port": each.port, "target": each.target} for each in self.bindings],
            "endpoints": [
                {"unit": each.unit, "bus": each.bus, "offset": each.offset, "range": each.range}
                for each in self.endpoints
            ],
            "clocks": [
                {"unit": each.unit, "pin": each.pin, "reset": each.reset, "clock": each.clock}
                for each in self.clocks
            ],
        }


@dataclass(frozen=True)
class Linked:
    """What a linker made of a description: the ``image`` the platform loads, the
    ``placement`` (each endpoint, ``<unit>/<bus>``, to its base address, where the platform
    has no name lookup) and its ``reports`` by name."""

    image: Path
    placement: Mapping[str, int]
    reports: Mapping[str, Path]


class Linker(Protocol):
    """A declared linker's code: it links a description on a board (the board facts its
    declaration names, ``board_keys``) into ``directory``, every tool run through
    ``tools``."""

    def link(
        self,
        integration: Integration,
        board: Mapping[str, str],
        directory: Path,
        tools: Toolchain,
    ) -> Linked: ...


class IntegrationError(Refusal):
    """A build the description refuses, or a linker refuses to link: one rejection, owned
    by who refused (``describe``, the linker, or the target's shell), its message the
    finding's line."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),))


def _range(bus: Bus, least: int) -> int:
    """The bytes an AXI-Lite ``bus`` takes: ``2**awaddr``, and ``least`` at least. A bus with
    no ``awaddr`` is refused, naming it."""
    widths = [signal.width for signal in bus.signals if signal.logical == "awaddr"]
    if not widths:
        raise IntegrationError(
            "bus-aperture-unknown",
            "describe",
            f"{bus.name}: an AXI-Lite bus with no awaddr has no aperture",
        )
    return max(1 << widths[0], least)


#: A mover's pins by their role: its stream, its memory port and its control bus.
MOVER_ROLES = {
    StandardProtocol.AXIS: "stream",
    StandardProtocol.AXIMM: "memory",
    StandardProtocol.AXILITE: "control",
}


def mover_pins(unit: str, module: Leaf) -> dict[str, Bus]:
    """The pins of the mover ``unit``'s ``module`` by role (``MOVER_ROLES``): one stream,
    one memory port and one control bus, each a bus of its module's. A module of other
    pins is refused (``mover-pins-unsupported``), naming the unit."""
    found: dict[str, list[Bus]] = {role: [] for role in MOVER_ROLES.values()}
    for pin in module.abi.pins:
        if isinstance(pin, Bus) and isinstance(pin.protocol, StandardProtocol):
            found[MOVER_ROLES[pin.protocol]].append(pin)
    if any(len(buses) != 1 for buses in found.values()):
        counts = ", ".join(f"{len(buses)} {role}" for role, buses in found.items())
        raise IntegrationError(
            "mover-pins-unsupported",
            "describe",
            f"{unit}: its module {module.name} presents {counts} buses; the description "
            "connects one stream, one memory port and one control bus a mover",
        )
    return {role: buses[0] for role, buses in found.items()}


def describe(model: ModelWrapper) -> Integration:
    """The integration description of the build's parent graph ``model``: its partition's
    network, as emission builds it (``configured_network``), the design its check verified
    (``refuse_unverified``), in its shell (see the module docstring)."""
    node, body, _ = partition_body(model)
    refuse_unverified(body, node.name)
    point, boundary = configured_network(body, node.name)
    row = point.row
    if row.linker is None:
        raise IntegrationError(
            "integration-unexported",
            "configuration.target.shell",
            f"the {row.shell!r} shell has no linker: its user links the core",
        )
    least = 1 if row.control is None else row.control.least_window
    offer = placed_offer(row)
    contracts = mover_contracts(point.network)
    contracted = dict(contracts.value) if isinstance(contracts, Available) else {}
    inputs: list[Supply] = []
    outputs: list[Delivery] = []
    for tensor, port in boundary:
        unit = point.network.mover(tensor)
        if unit is None or unit not in contracted or offer is None:
            raise IntegrationError(
                "mover-absent",
                "describe",
                f"{tensor} ({port}): the {row.shell!r} shell places no mover on it; its "
                "integration connects every boundary port to a mover",
            )
        shape = known_shape(body, tensor)
        if shape is None:
            raise IntegrationError(
                "shape-unstated",
                "describe",
                f"{tensor} ({port}): the partition states no shape for it",
            )
        contract = contracted[unit]
        edge = (Supply if contract.direction == "in" else Delivery)(
            port=port,
            tensor=tensor,
            shape=shape,
            element=contract.element,
            kind=offer.kind,
            unit=unit,
            target=row.memory[0].name,
            contract=contract,
            module=library_module(point.network.member(unit)),
        )
        if isinstance(edge, Supply):
            inputs.append(edge)
        else:
            outputs.append(edge)
    core = Core(name=node.name, inputs=tuple(inputs), outputs=tuple(outputs))
    buses = [
        pin
        for pin in point.module.abi.pins
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXILITE
    ]
    ip = IpRef(vlnv(module_name(point.module, source_roots())), ())
    units = (
        *(Unit(name=each.unit, kind=each.contract.kind, ip=None, window=least) for each in inputs),
        Unit(name=core.name, kind=CORE, ip=ip, window=None),
        *(Unit(name=each.unit, kind=each.contract.kind, ip=None, window=least) for each in outputs),
    )
    names = [unit.name for unit in units]
    shared = sorted({name for name in names if names.count(name) > 1})
    if shared:
        raise IntegrationError(
            "unit-name-collides",
            "describe",
            f"the core {core.name!r} is named as a mover's unit ({', '.join(shared)}): "
            "each unit is named once",
        )
    pins = {each.unit: mover_pins(each.unit, each.module) for each in (*inputs, *outputs)}
    connections = (
        *(
            Connection(
                f"{each.unit}/{pins[each.unit]['stream'].name}",
                f"{core.name}/{each.port}",
                each.contract.tdata,
            )
            for each in inputs
        ),
        *(
            Connection(
                f"{core.name}/{each.port}",
                f"{each.unit}/{pins[each.unit]['stream'].name}",
                each.contract.tdata,
            )
            for each in outputs
        ),
    )
    endpoints: list[Endpoint] = []
    for placed in units:
        controls = buses if placed.kind == CORE else [pins[placed.name]["control"]]
        endpoints += [
            Endpoint(unit=placed.name, bus=bus.name, offset=0, range=_range(bus, least))
            for bus in controls
        ]
    target = read_target(body)
    return Integration(
        row=(row.shell, row.board),
        linker=row.linker,
        runtime=row.runtimes[0] if row.runtimes else None,
        part=target.part,
        period_ns=target.platform.period_ns,
        cores=(core,),
        units=units,
        connections=connections,
        bindings=tuple(
            Binding(f"{each.unit}/{pins[each.unit]['memory'].name}", each.target)
            for each in (*inputs, *outputs)
        ),
        endpoints=tuple(endpoints),
        clocks=tuple(ClockPin(placed.name, CLOCK, RESET, MAIN) for placed in units),
    )


__all__ = [
    "CLOCK",
    "CORE",
    "MAIN",
    "MOVER_ROLES",
    "RESET",
    "Binding",
    "ClockPin",
    "Connection",
    "Core",
    "Delivery",
    "Edge",
    "Endpoint",
    "Integration",
    "IntegrationError",
    "IpRef",
    "Linked",
    "Linker",
    "Supply",
    "Unit",
    "describe",
    "mover_pins",
]
