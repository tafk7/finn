# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as nets: the stream interface, its link family and the netlist fold.

A kernel declares each stream endpoint as a ``Port(STREAM, direction,
offer=<contract view>)``. A composite declares one ``Net(StreamLink)`` (or
``BufferedStreamLink``) per stream and attaches ports to it: children at
placement (``Subspace(Kernel, input_stream=net)``), and the composite's own
ports from inside (``Port(STREAM, "in", net=net)``), which is how a child's
endpoint becomes a boundary port of the composed module.

The carried value is the ``StreamSpec``. Some end, or the parent through
``carry=``, publishes it; the others adopt it. Each end offers its
``StreamContract``; a boundary end's contract is the AXIS port this module
presents. Each net owns its ``compatible`` constraint and its accepted
``connection`` view, so a refusal belongs to that net. ``NETLIST`` folds a
composite's accepted modules and connections into one wrapper module; instance
names come from the declaration names of the nodes and nets.

A ``BufferedStreamLink`` owns a ``transport`` choice between ``direct`` and
``fifo``. The FIFO case owns its ``depth`` and the FIFO's ``ram_style``; it is
an identity adapter, checked on both of its sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finn.core.space import (
    Carried,
    Decision,
    Ends,
    Interface,
    Interpretation,
    Link,
    Param,
    Rejected,
    Space,
    Subspace,
    SubspaceChoice,
    Topology,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    domain,
    reject,
    view,
)
from finn.core.space.graph import End
from finn.kernels.artifacts.abi import Clock, ClockAlignment, Direction, Endpoint, Reset, Signal
from finn.kernels.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.fifo import FifoKernel
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import (
    STREAM_CONTRACT,
    Mismatch,
    StreamContract,
    compatibility,
)
from finn.kernels.physical.forms import Every, Repetition, Traversal
from finn.kernels.physical.lowering import lower_module_structure
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.physical.validation import abi_pins


@dataclass(frozen=True)
class StreamSpec:
    """The logical sequence a stream carries per consumer pass."""

    element: ScalarEncoding
    form: Traversal
    repetition: Repetition = Repetition.ONCE
    markers: tuple[Every, ...] = ()

    @property
    def payload_bits(self) -> int:
        return self.form.lanes * self.element.bits


STREAM_SPEC = default_semantics(StreamSpec)


STREAM: Interface[StreamSpec, StreamContract] = Interface(
    "stream", carries=STREAM_SPEC, offers=STREAM_CONTRACT
)
MODULE = ViewKey("module", default_semantics(ModuleBuildRequirements))
_CLOCKING = {"clock": "ap_clk", "reset": "ap_rst_n"}


def boundary_contract(name: str, spec: StreamSpec, endpoint: Endpoint) -> StreamContract:
    """The AXIS port a composed module presents for one of its own stream ports."""
    if len(spec.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    stream = AxiStream(
        name, spec.element.dtype, spec.form.lanes, endpoint=endpoint, last=bool(spec.markers)
    )
    transport = stream.native(**_CLOCKING)
    markers = {transport.markers[0].signal: spec.markers[0]} if spec.markers else {}
    return StreamContract(transport, spec.element, spec.form, spec.repetition, markers)


@dataclass(frozen=True)
class Stage:
    """A transport stage inside a stream: none (direct) or a module with two ports."""

    requirements: ModuleBuildRequirements | None = None
    input: StreamContract | None = None
    output: StreamContract | None = None


STAGE_SEMANTICS = default_semantics(Stage)
STAGE = ViewKey("stage", STAGE_SEMANTICS)


class _Direct(Space):
    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()

    exports = {STAGE: stage}


class StreamFifo(Space):
    """An identity adapter: the stream's own spec on both sides of a native FIFO."""

    spec = Param(STREAM_SPEC)

    @derived
    def word_bits(self) -> int:
        return self.spec.payload_bits

    buffer = Subspace(
        FifoKernel,
        word_bits=word_bits,
        depth=Decision(int, domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        spec = self.spec
        source, sink = self.buffer.interfaces()
        return Stage(
            self.buffer.build_requirements(),
            StreamContract(source, spec.element, spec.form),
            StreamContract(sink, spec.element, spec.form),
        )

    exports = {STAGE: stage}


@dataclass(frozen=True)
class Connection:
    """One checked stream; an owner of None is the composed module's own port."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract
    stage: Stage


CONNECTION = default_semantics(Connection)
CONNECTION_VIEW = ViewKey("connection", CONNECTION)


@dataclass(frozen=True)
class Endpoints:
    source: End[StreamContract]
    source_contract: StreamContract
    sink: End[StreamContract]
    sink_contract: StreamContract


ENDPOINTS = default_semantics(Endpoints)


def _contract(end: End[StreamContract], spec: StreamSpec) -> StreamContract:
    if end.node is not None:
        assert end.offer is not None
        return end.offer
    # Inside, the composite's input port drives the net: it is the AXIS target.
    endpoint = Endpoint.TARGET if end.direction == "out" else Endpoint.INITIATOR
    return boundary_contract(end.port, spec, endpoint)


class StreamLink(Link):
    """One stream: a single producer and a single consumer (no data fan-out)."""

    spec = Carried(STREAM)
    ends = Ends(STREAM)

    @derived(semantics=ENDPOINTS)
    def endpoints(self) -> Endpoints | Rejected:
        sources = [end for end in self.ends if end.direction == "out"]
        sinks = [end for end in self.ends if end.direction == "in"]
        if len(sources) != 1 or len(sinks) != 1:
            return reject(
                "stream-ends",
                f"a stream joins one producer and one consumer, not {len(sources)} and "
                f"{len(sinks)}",
            )
        spec = self.spec
        try:
            return Endpoints(
                sources[0], _contract(sources[0], spec), sinks[0], _contract(sinks[0], spec)
            )
        except ValueError as error:
            return reject("stream-boundary", str(error))

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()

    @constraint
    def compatible(self) -> bool | Rejected:
        ends, stage = self.endpoints, self.stage()
        source, sink = ends.source_contract, ends.sink_contract
        source_top, sink_top = ends.source.node is None, ends.sink.node is None
        if stage.requirements is None:
            found = list(
                compatibility(source, sink, source_is_top=source_top, sink_is_top=sink_top)
            )
        else:
            assert stage.input is not None and stage.output is not None
            found = [
                *compatibility(source, stage.input, source_is_top=source_top, sink_is_top=False),
                *compatibility(stage.output, sink, source_is_top=False, sink_is_top=sink_top),
            ]
        return _refusal(found)

    @derived(semantics=CONNECTION)
    def link(self) -> Connection:
        ends = self.endpoints
        return Connection(
            ends.source.node, ends.source_contract, ends.sink.node, ends.sink_contract, self.stage()
        )

    connection = View(link, constraints=(compatible,))
    exports = {CONNECTION_VIEW: connection}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


class BufferedStreamLink(StreamLink):
    transport = SubspaceChoice(
        {"direct": Subspace(_Direct), "fifo": Subspace(StreamFifo, spec=StreamLink.spec)},
        exports=(STAGE,),
    )
    stage = View(transport.accepted(STAGE))


@dataclass(frozen=True)
class Composed:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements


COMPOSED = default_semantics(Composed)


def _instance(node: str | None) -> str | None:
    return None if node is None else "u_" + node


def netlist(
    *,
    topology: Topology[ModuleBuildRequirements, Connection],
    module: str,
    producer: ProducerIdentity,
) -> Composed | Rejected:
    """Wire a composite's accepted modules through its accepted streams.

    Each node's module is instantiated as ``u_<node>``, each FIFO stage as
    ``u_<net>_fifo``. A defect no net can see alone, such as a clock-domain
    conflict, is refused here.
    """
    try:
        return _netlist(topology, module, producer)
    except ValueError as error:
        return reject("stream-composition", str(error))


def _netlist(
    topology: Topology[ModuleBuildRequirements, Connection],
    module: str,
    producer: ProducerIdentity,
) -> Composed:
    placed = {"u_" + name: requirements for name, requirements in topology.nodes}
    connections = [
        (
            net.name,
            replace(
                net.value,
                source_owner=_instance(net.value.source_owner),
                sink_owner=_instance(net.value.sink_owner),
            ),
        )
        for net in topology.nets
    ]
    fifos = {name: c.stage for name, c in connections if c.stage.requirements is not None}
    boundary = [c.source for _, c in connections if c.source_owner is None] + [
        c.sink for _, c in connections if c.sink_owner is None
    ]
    children = [*placed.values(), *(s.requirements for s in fifos.values() if s.requirements)]
    composition = Composition(_top_abi(module, boundary, children))
    stream_pins: dict[str, set[str]] = {}
    for _, c in connections:
        for owner, contract in ((c.source_owner, c.source), (c.sink_owner, c.sink)):
            if owner is not None:
                stream_pins.setdefault(owner, set()).update(
                    pin.name for pin in contract.transport.pins()
                )
    for name, requirements in placed.items():
        composition.add(name, requirements)
        _drive(composition, name, requirements, stream_pins.get(name, set()))
    for stream, stage in fifos.items():
        assert stage.requirements and stage.input and stage.output
        instance = f"u_{stream}_fifo"
        composition.add(instance, stage.requirements)
        pins = {p.name for p in (*stage.input.transport.pins(), *stage.output.transport.pins())}
        _drive(composition, instance, stage.requirements, pins)
    for name, c in connections:
        source, sink = StreamEnd(c.source_owner, c.source), StreamEnd(c.sink_owner, c.sink)
        if c.stage.input is not None and c.stage.output is not None:
            instance = f"u_{name}_fifo"
            composition.connect(source, StreamEnd(instance, c.stage.input))
            source = StreamEnd(instance, c.stage.output)
        composition.connect(source, sink)
    structure = composition.finish()
    wrapper = RenderedSourceRequirement(
        EntryPointSourceName(),
        "decomposed_wrapper.sv.j2",
        ("PORT_DECLARATIONS", "NET_DECLARATIONS", "ASSIGNMENTS", "INSTANCES"),
        SELF_CONTAINED_JINJA_RENDERER,
        requires=tuple(
            "module:" + instance.requirements.abi.entry_point.value
            for instance in structure.instances
            if isinstance(instance.requirements.abi.entry_point, FixedModuleName)
        ),
        provides_entry_point=True,
    )
    return Composed(
        structure, lower_module_structure(structure, producer=producer, wrapper_template=wrapper)
    )


NETLIST: Interpretation[ModuleBuildRequirements, Connection, Composed] = Interpretation(
    "netlist", node=MODULE, net=CONNECTION_VIEW, result=COMPOSED, reduce=netlist
)


def _top_abi(
    module: str, boundary: list[StreamContract], children: list[ModuleBuildRequirements]
) -> ModuleABIRequirements:
    """Top clocks and reset follow the children's own AXIS clocking; boundary buses follow."""
    signals: dict[str, Signal] = {}
    alignments: list[ClockAlignment] = []
    for requirements in children:
        for port in requirements.abi.ports:
            if isinstance(port, Signal) and port.name in ("ap_clk", "ap_clk2x", "ap_rst_n"):
                signals.setdefault(port.name, port)
        for alignment in requirements.abi.clock_alignments:
            if alignment not in alignments:
                alignments.append(alignment)
    signals.setdefault("ap_clk", Signal("ap_clk", Direction.IN, 1, Clock()))
    signals.setdefault(
        "ap_rst_n",
        Signal(
            "ap_rst_n",
            Direction.IN,
            1,
            Reset(active_low=True, synchronous=True, synchronous_to=("ap_clk",)),
        ),
    )
    clocking = tuple(
        signals[name] for name in ("ap_clk", "ap_clk2x", "ap_rst_n") if name in signals
    )
    return ModuleABIRequirements(
        GeneratedModuleName(module),
        (*clocking, *(contract.transport.axis_bus() for contract in boundary)),
        (),
        tuple(alignments),
    )


def _drive(
    composition: Composition,
    instance: str,
    requirements: ModuleBuildRequirements,
    stream_pins: set[str],
) -> None:
    """Route every input outside the instance's streams: clocks and resets by role."""
    for name, info in abi_pins(requirements.abi).items():
        if info.bus_id is not None or info.direction is not Direction.IN or name in stream_pins:
            continue
        if "clk2x" in name:
            composition.drive(instance, name, "ap_clk2x")
        elif isinstance(info.role, Clock):
            composition.drive(instance, name, "ap_clk")
        elif isinstance(info.role, Reset):
            composition.drive(instance, name, "ap_rst_n")
        else:
            raise ValueError(f"{instance}.{name}: an input outside every stream has no driver")


__all__ = [
    "BufferedStreamLink",
    "COMPOSED",
    "CONNECTION",
    "Composed",
    "Connection",
    "MODULE",
    "NETLIST",
    "STREAM",
    "STREAM_SPEC",
    "Stage",
    "StreamFifo",
    "StreamLink",
    "StreamSpec",
    "boundary_contract",
    "netlist",
]
