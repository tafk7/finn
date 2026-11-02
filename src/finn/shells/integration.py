# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The integration description: what a shell's linker builds around a build's cores, with no
platform's names in it, and what a linker gives back.

``describe(model, runtime)`` takes the build's parent graph (its one partition node,
``finn.core.graph.partition.partition_body``), refuses a body whose design its check did not
verify as it is now (``refuse_unverified``), configures the partition's network as hardware
generation does (``configured_network``), and reads the movers its shell places
(``finn.kernels.shell``, each its contract, ``IodmaKernel.contract``, and its module) and its
shell's row. It states (``Integration``):

- the row it was made for, its linker, its runtime (the build's, of those the row offers:
  the row's first unless named; ``runtime-not-offered``), the part and the period asked;
- each core (``Core``): its node's name, and per edge port its supply (``Supply``) or
  delivery (``Delivery``): the tensor, its shape and element, the offer's kind, the mover's
  unit, the memory target it binds, the mover's facts (``MoverContract``) and its module,
  whose IP its unit is;
- the units the platform places (``Unit``): each core's, named as its node, and each
  mover's, ``in<i>`` and ``out<j>``, its name in the shell (by its direction and its edge
  port's index), each with its kind, its IP (``IpRef``) and its control window; each
  named once, as its shell names it (``finn.kernels.shell``);
- the connections (``Connection``): each mover's stream to its core's edge port, with its
  bits;
- the bindings (``Binding``): each mover's memory port to a memory target of the row;
- the endpoints (``Endpoint``): each control bus, by unit and bus, at its offset in its unit
  and its range: a core's bus with the KernelOp node presenting it (``node``), a mover's
  with the registers a runtime drives it by (``registers``, its contract's);
- the clocks (``ClockPin``): each unit's clock and reset pins, on a clock the row names.

Every port is its unit's own pin: a core's edge ports its module's, a mover's stream,
memory port and control bus its module's (``mover_pins``); a unit with a window is reached
by one control port, ``s_axi_control`` (``finn.kernels.artifacts.unit.CONTROL``), a mover's
its control bus. Where each unit is placed and what its cell is called are the linker's
(``Linker``): ``finn.shells.pynq.linker`` for the ``pynq`` shell. A mover's IP is its
module, built by the build of the bitfile (``finn.shells.build``; its unit's ``ip`` is
``None`` until then). A core
presenting no control bus is its unit, its IP the one its packaging gave it (its body's
``finn.outputs`` ``vlnv`` and ``ip``): nothing predicts it, and before packaging there is
none. A core presenting buses is placed as its core unit (``finn.kernels.artifacts.unit``):
the core and its grouper (``core_grouping``), its window the grouper's, its IP built by the
build from the core's (``None`` until then); each of its buses an endpoint at its offset in
the grouper's map, of its own range, ``2**awaddr`` bytes. A bus the grouper cannot take, a
32-bit AXI-Lite target's members, is refused (``grouper-bus-unsupported``), naming it.

Each mover binds the row's first memory target. A mover's control bus is its one endpoint,
at offset 0, of its own range and the row's least window at least, its unit's window that
range. Every endpoint takes a byte at least, at an offset of zero or more
(``endpoint-invalid``), and lies in its unit's window, where its unit has one
(``endpoint-outside-unit``).

It is a function of a configured point: the description is a build's, needed once a build
and not at every point an exploration costs. A shell without a linker (``ip``) has no
description: its user links the core.

The runtime file (``runtime_file``, written as ``runtime_text``, ``integration.json``,
``RUNTIME_SCHEMA``) is the description as every runtime reads it (``Integration.report``: no
unit's IP, which is the build's), with what its linker made of it (``Linked``): each placed
unit's base, where the platform has no name lookup, each clock as delivered, and the
files the image loads with. A declared runtime's code (``Runtime``) writes the driver that
reads it on the board.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from finn.core.graph.annotations.outputs import OUTPUT_IP, OUTPUT_VLNV
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.facts import known_shape
from finn.core.graph.partition import core_bodies
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import Available
from finn.custom_op.kernels.shell import (
    Configured,
    build_shell,
    configured_network,
    core_grouping,
    refuse_unverified,
)
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.artifacts.abi import Bus, StandardProtocol, bus_window
from finn.kernels.artifacts.module import Leaf, Registers
from finn.kernels.artifacts.unit import CONTROL, unsupported
from finn.kernels.composer import member_name
from finn.kernels.library import library_module
from finn.kernels.movers import MoverContract
from finn.kernels.shell import ShellKernel
from finn.platform.shells import Offer, placed_offer

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.toolchain.launcher import Toolchain

CORE = "core"
"""A core unit's kind."""

MAIN = "main"
"""The row's main clock (``finn.platform.shells.Clocks``)."""

CLOCK, RESET = "ap_clk", "ap_rst_n"
"""Every unit's clock and reset pins."""

RUNTIME_SCHEMA = 1
"""The layout of the runtime file (``runtime_file``), its ``schema``."""


@dataclass(frozen=True)
class IpRef:
    """A unit's IP: its ``vlnv`` and the IP ``repositories`` a tool adds to find it."""

    vlnv: str
    repositories: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Unit:
    """What the platform places: its ``name``, its ``kind`` (``core``, or a mover's kind),
    its ``ip`` (``None`` until it is built: a core's by its packaging, a mover's by the
    bitfile's build) and the bytes of its control
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
    """A control bus: its ``unit``, the ``bus`` inside it, the graph ``node`` it configures
    (a core's bus: the KernelOp presenting it; None for a mover's), at ``offset`` in the
    unit's window, of ``range`` bytes, and the ``registers`` a runtime drives it by (a
    mover's; none for a core's, whose writes are its node's configuration)."""

    unit: str
    bus: str
    node: str | None = None
    offset: int
    range: int
    registers: Registers = Registers()


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
    docstring). Each endpoint is in its unit's control window: one at a negative offset or
    of no byte (``endpoint-invalid``), of no unit or of a unit with no window
    (``endpoint-unplaced``), or reaching past its unit's window (``endpoint-outside-unit``)
    is refused."""

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

    def __post_init__(self) -> None:
        windows = {unit.name: unit.window for unit in self.units}
        for each in self.endpoints:
            if each.offset < 0 or each.range < 1:
                raise IntegrationError(
                    "endpoint-invalid",
                    "describe",
                    f"{each.unit}/{each.bus}: {each.range} bytes at offset {each.offset}; an "
                    "endpoint takes a byte at least, at an offset of zero or more",
                )
            if each.unit not in windows or windows[each.unit] is None:
                raise IntegrationError(
                    "endpoint-unplaced",
                    "describe",
                    f"{each.unit}/{each.bus}: "
                    + (
                        "no unit of that name"
                        if each.unit not in windows
                        else "its unit has no control window to place it in"
                    ),
                )
            window = windows[each.unit]
            if window is not None and each.offset + each.range > window:
                raise IntegrationError(
                    "endpoint-outside-unit",
                    "describe",
                    f"{each.unit}/{each.bus}: {each.range} bytes at offset {each.offset} lie "
                    f"outside its unit's window of {window}",
                )

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
        """The description as JSON, as a runtime reads it: each edge with its mover's
        stream (``bits`` a beat, ``beats`` a frame, ``lanes`` a beat), each endpoint with its
        node and its registers; no unit's IP, which is the build's."""

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
                "bits": each.contract.tdata,
                "beats": each.contract.beats,
                "lanes": each.contract.lanes,
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
                {"name": unit.name, "kind": unit.kind, "window": unit.window} for unit in self.units
            ],
            "connections": [
                {"source": each.source, "sink": each.sink, "bits": each.bits}
                for each in self.connections
            ],
            "bindings": [{"port": each.port, "target": each.target} for each in self.bindings],
            "endpoints": [
                {
                    "unit": each.unit,
                    "bus": each.bus,
                    "node": each.node,
                    "offset": each.offset,
                    "range": each.range,
                    "registers": {
                        name: {"offset": offset, "bits": bits}
                        for name, offset, bits in each.registers.registers
                    },
                }
                for each in self.endpoints
            ],
            "clocks": [
                {"unit": each.unit, "pin": each.pin, "reset": each.reset, "clock": each.clock}
                for each in self.clocks
            ],
        }


@dataclass(frozen=True)
class Timing:
    """A linked image's routed timing, as its linker read it from its routed timing report
    and admitted it: the period its main clock closed at (``period_ns``), the routed
    design's worst setup slack there (``wns_ns``, never negative: a linker refuses an image
    that misses timing) and the report it read (``report``, its path under a build's
    ``report/``, a key of ``Linked.reports``)."""

    period_ns: float
    wns_ns: float
    report: str

    def record(self) -> dict[str, Any]:
        """The timing as JSON."""
        return {"period_ns": self.period_ns, "wns_ns": self.wns_ns, "report": self.report}


@dataclass(frozen=True)
class Linked:
    """What a linker made of a description: the ``image`` the platform loads, the
    ``placement`` (each unit with a control window to its base address, where the platform
    has no name lookup; an endpoint is at its unit's base and its offset), each of the row's
    clocks as it delivers it, in MHz (``clocks_mhz``: none it does not know), its
    ``reports``, each by its path under a build's ``report/`` (a file, or a directory of
    them), the files the image loads with (``beside``: PYNQ's hardware handoff beside
    its bitfile) and its routed timing as admitted (``timing``: None where the linker
    states none, an image with no routed design)."""

    image: Path
    placement: Mapping[str, int]
    clocks_mhz: Mapping[str, float]
    reports: Mapping[str, Path]
    beside: tuple[Path, ...] = ()
    timing: Timing | None = None


def runtime_file(integration: Integration, linked: Linked) -> dict[str, Any]:
    """The runtime file of ``integration`` as ``linked`` made it (``integration.json``):
    its layout (``schema``, ``RUNTIME_SCHEMA``), the description as a runtime reads it
    (``Integration.report``: ``period_ns`` the build's request among it), each placed unit's
    base (``placement``) and each clock as delivered (``clocks_mhz``)."""
    return {
        "schema": RUNTIME_SCHEMA,
        **integration.report(),
        "placement": dict(linked.placement),
        "clocks_mhz": dict(linked.clocks_mhz),
    }


def runtime_text(integration: Integration, linked: Linked) -> str:
    """The runtime file of ``integration`` as ``linked`` made it (``runtime_file``), as
    written: the same text in the build's report and beside its driver."""
    return json.dumps(runtime_file(integration, linked), indent=2)


class Linker(Protocol):
    """A declared linker's code, made with the build's ``debug`` option (its debug
    instrumentation, where it has any): it refuses a description it cannot link on a board
    (the board facts its declaration names, ``board_keys``) before any tool runs
    (``check``), and links one, every unit's IP given, into ``directory``, every tool run
    through ``tools`` (its parallel runs the toolchain's selection's)."""

    def check(self, integration: Integration, board: Mapping[str, str]) -> None: ...

    def link(
        self,
        integration: Integration,
        board: Mapping[str, str],
        directory: Path,
        tools: Toolchain,
    ) -> Linked: ...


class Runtime(Protocol):
    """A declared runtime's code: it writes what runs a build on its board, from its
    description and what its linker made of it (``linked``), into ``directory``: its driver
    and the runtime file beside it."""

    def emit(self, integration: Integration, linked: Linked, directory: Path) -> None: ...


class IntegrationError(Refusal):
    """A build the description refuses, or a linker refuses to link: one rejection, owned
    by who refused (``describe``, the linker, or the target's shell), its message the
    finding's line."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),))


def _range(bus: Bus, least: int) -> int:
    """The bytes an AXI-Lite ``bus`` takes: ``2**awaddr``, and ``least`` at least. A bus with
    no ``awaddr`` is refused, naming it."""
    window = bus_window(bus)
    if window is None:
        raise IntegrationError(
            "bus-aperture-unknown",
            "describe",
            f"{bus.name}: an AXI-Lite bus with no awaddr has no aperture",
        )
    return max(window, least)


#: A mover's pins by their role: its stream, its memory port and its control bus.
MOVER_ROLES = {
    StandardProtocol.AXIS: "stream",
    StandardProtocol.AXIMM: "memory",
    StandardProtocol.AXILITE: "control",
}


def mover_pins(unit: str, module: Leaf) -> dict[str, Bus]:
    """The pins of the mover ``unit``'s ``module`` by role (``MOVER_ROLES``): one stream,
    one memory port and one control bus, each a bus of its module's, the control bus its
    unit's port (``CONTROL``). A module of other pins is refused
    (``mover-pins-unsupported``), naming the unit."""
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
    if found["control"][0].name != CONTROL:
        raise IntegrationError(
            "mover-pins-unsupported",
            "describe",
            f"{unit}: its module {module.name}'s control bus is {found['control'][0].name}, "
            f"not its unit's port, {CONTROL}",
        )
    return {role: buses[0] for role, buses in found.items()}


def describe(model: ModelWrapper, runtime: str | None = None) -> Integration:
    """The integration description of the build's parent graph ``model``: every
    DataflowCore of it (``core_bodies``), in its order, each the design its check verified
    (``refuse_unverified``), its network as emission builds it (``configured_network``),
    all in the build's shell (``build_shell``: its movers named across the cores, its rules
    over every unit), in its shell, for the build's ``runtime`` (the row's first unless
    named; one the row does not offer refused, ``runtime-not-offered``; see the module
    docstring)."""
    configured = []
    for node, body, _ in core_bodies(model):
        refuse_unverified(body, node.name)
        configured.append((node, body, configured_network(body, node.name)))
    row = configured[0][2].row
    if row.linker is None:
        raise IntegrationError(
            "integration-unexported",
            "configuration.target.shell",
            f"the {row.shell!r} shell has no linker: its user links the core",
        )
    if runtime is None:
        runtime = row.runtimes[0] if row.runtimes else None
    elif runtime not in row.runtimes:
        raise IntegrationError(
            "runtime-not-offered",
            "configuration.target.runtime",
            f"the {row.shell!r} shell does not offer the runtime {runtime!r} "
            f"(it offers {list(row.runtimes)})",
        )
    least = 1 if row.control is None else row.control.least_window
    offer = placed_offer(row)
    whole = build_shell(model)
    placed = {edge: unit for unit, edge in type(whole).movers.items()}
    cores: list[Core] = []
    units: list[Unit] = []
    connections: list[Connection] = []
    endpoints: list[Endpoint] = []
    for node, body, point in configured:
        core, core_units, core_connections, core_endpoints = _core(
            node.name, body, point, whole, placed, offer, least
        )
        cores.append(core)
        units += core_units
        connections += core_connections
        endpoints += core_endpoints
    target = read_target(configured[0][1])
    edges = [edge for core in cores for edge in (*core.inputs, *core.outputs)]
    return Integration(
        row=(row.shell, row.board),
        linker=row.linker,
        runtime=runtime,
        part=target.part,
        period_ns=target.platform.period_ns,
        cores=tuple(cores),
        units=tuple(units),
        connections=tuple(connections),
        bindings=tuple(
            Binding(f"{each.unit}/{mover_pins(each.unit, each.module)['memory'].name}", each.target)
            for each in edges
        ),
        endpoints=tuple(endpoints),
        clocks=tuple(ClockPin(unit.name, CLOCK, RESET, MAIN) for unit in units),
    )


def _core(
    name: str,
    body: ModelWrapper,
    point: Configured,
    whole: ShellKernel,
    placed: Mapping[tuple[str, str], str],
    offer: Offer | None,
    least: int,
) -> tuple[Core, list[Unit], list[Connection], list[Endpoint]]:
    """Core ``name`` of the build (its body ``body``, its network as emission builds it,
    ``point``) in the build's shell ``whole``, whose movers serve each edge port
    (``placed``: by the core's member and the port): the core with its edge's supplies and
    deliveries, its units (its input movers', its own, its output movers'), its movers'
    streams to its edge ports, and its endpoints, each mover's control bus at offset 0 of
    ``least`` bytes at least."""
    row = point.row
    inputs: list[Supply] = []
    outputs: list[Delivery] = []
    for tensor, port in point.network.boundary:
        unit = placed.get((member_name(name), port))
        member = None if unit is None else getattr(whole, unit)
        contract = None if member is None else member.query(type(member).contract)
        if unit is None or member is None or offer is None or not isinstance(contract, Available):
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
        stated: MoverContract = contract.value
        edge = (Supply if stated.direction == "in" else Delivery)(
            port=port,
            tensor=tensor,
            shape=shape,
            element=stated.element,
            kind=offer.kind,
            unit=unit,
            target=row.memory[0].name,
            contract=stated,
            module=library_module(member),
        )
        if isinstance(edge, Supply):
            inputs.append(edge)
        else:
            outputs.append(edge)
    core = Core(name=name, inputs=tuple(inputs), outputs=tuple(outputs))
    buses = [
        pin
        for pin in point.module.abi.pins
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXILITE
    ]
    grouping = core_grouping(point.network)
    for bus in buses if grouping is not None else ():
        reason = unsupported(bus)
        if reason is not None:
            raise IntegrationError(
                "grouper-bus-unsupported",
                "describe",
                f"{core.name}: {reason}; its unit's grouper takes 32-bit AXI-Lite targets",
            )
    # Each bus is presented by its member's instance, named by the member's node.
    owners = {member: graph for graph, member in point.network.names.members.items()}
    presenting = {
        each.port: owners[each.instance.split(".")[0]] for each in point.module.fragment.exports
    }
    packaged, packaged_ip = body.get(OUTPUT_VLNV), body.get(OUTPUT_IP)
    ip = None if packaged is None or packaged_ip is None else IpRef(packaged, (packaged_ip,))
    pins = {each.unit: mover_pins(each.unit, each.module) for each in (*inputs, *outputs)}

    def mover(edge: Edge) -> Unit:
        window = _range(pins[edge.unit]["control"], least)
        return Unit(name=edge.unit, kind=edge.contract.kind, ip=None, window=window)

    placed_core = (
        Unit(name=core.name, kind=CORE, ip=ip, window=None)
        if grouping is None
        else Unit(name=core.name, kind=CORE, ip=None, window=grouping[1])
    )
    units = [*map(mover, inputs), placed_core, *map(mover, outputs)]
    connections = [
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
    ]
    contracts = {each.unit: each.contract for each in (*inputs, *outputs)}
    endpoints: list[Endpoint] = []
    for each in units:
        if each.kind != CORE:
            control = pins[each.name]["control"]
            endpoints.append(
                Endpoint(
                    unit=each.name,
                    bus=control.name,
                    offset=0,
                    range=_range(control, least),
                    registers=contracts[each.name].registers,
                )
            )
        elif grouping is not None:
            endpoints += [
                Endpoint(
                    unit=each.name,
                    bus=bus.name,
                    node=presenting[bus.name],
                    offset=offset,
                    range=_range(bus, 1),
                )
                for bus, offset in zip(buses, grouping[0], strict=True)
            ]
    return core, units, connections, endpoints


__all__ = [
    "CLOCK",
    "CORE",
    "MAIN",
    "MOVER_ROLES",
    "RESET",
    "RUNTIME_SCHEMA",
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
    "Runtime",
    "Supply",
    "Timing",
    "Unit",
    "describe",
    "mover_pins",
    "runtime_file",
    "runtime_text",
]
