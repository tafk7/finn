# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declared streams: each connection is an ordinary part of the parent Space.

A parent derives a ``StreamSpec`` (element, traversal, repetition, markers) for
every connection, binds kernel ``Port`` formals to it, and declares a ``Stream``
naming its producer and consumer through accepted port views:

    weight_stream = Stream(
        weight_spec,
        source=("u_weights", implementation.accepted(OUTPUT_PORT)),
        sink=("u_compute", compute.accepted(DotpAxiKernel.weights_port)),
        buffered=True,
    )

Each stream owns its ``compatible`` constraint, so a refusal is attributed to
that stream and independent streams settle independently. Its accepted
``connection`` view is a detached ``Connection``. The parent's physical output is
a thin reduction: ``compose`` collects its instances and connections, then wires,
validates and lowers them. Nothing here discovers topology by inspection: the
linker owns every endpoint through explicit bindings.

A stream declared ``buffered=True`` owns a ``transport`` choice between
``direct`` and ``fifo``. The FIFO case owns its ``depth`` and the FIFO's
``ram_style``; it is an identity adapter, checked on both of its sides. Whether
a FIFO is needed and how deep is a compiler decision; the stream provides the slot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from finn.core.space import (
    Constraint,
    Decision,
    Param,
    Rejected,
    Space,
    Subspace,
    SubspaceChoice,
    ValueRef,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    domain,
    reject,
    view,
)
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


class Port(Param[StreamSpec]):
    """A kernel's stream formal: the parent binds the connection's spec.

    Optional, so a kernel can still be configured on its own without a stream.
    """

    def __init__(self, direction: Endpoint) -> None:
        super().__init__(STREAM_SPEC, required=False)
        self.direction = direction


@dataclass(frozen=True)
class Module:
    """A placement's module, or None for a boundary placement."""

    requirements: ModuleBuildRequirements | None


MODULE_SEMANTICS = default_semantics(Module)
MODULE = ViewKey("module", MODULE_SEMANTICS)
OUTPUT_PORT = ViewKey("output_port", STREAM_CONTRACT)
_CLOCKING = {"clock": "ap_clk", "reset": "ap_rst_n"}


def _top(name: str, spec: StreamSpec, endpoint: Endpoint) -> StreamContract:
    if len(spec.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    stream = AxiStream(
        name, spec.element.dtype, spec.form.lanes, endpoint=endpoint, last=bool(spec.markers)
    )
    transport = stream.native(**_CLOCKING)
    markers = {transport.markers[0].signal: spec.markers[0]} if spec.markers else {}
    return StreamContract(transport, spec.element, spec.form, spec.repetition, markers)


class TopInput(Space):
    """The composed module's own input port, producing a stream."""

    name = Param(str)
    output_stream = Port(Endpoint.INITIATOR)

    @view(semantics=STREAM_CONTRACT)
    def port(self) -> StreamContract:
        return _top(self.name, self.output_stream, Endpoint.TARGET)

    @view(semantics=MODULE_SEMANTICS)
    def module(self) -> Module:
        return Module(None)

    exports = {OUTPUT_PORT: port, MODULE: module}


class TopOutput(Space):
    """The composed module's own output port, consuming a stream."""

    name = Param(str)
    input_stream = Port(Endpoint.TARGET)

    @view(semantics=STREAM_CONTRACT)
    def port(self) -> StreamContract:
        return _top(self.name, self.input_stream, Endpoint.INITIATOR)


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
    """One checked stream; an owner of None is the composed module's boundary."""

    name: str
    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract
    stage: Stage


CONNECTION = default_semantics(Connection)


def _is_top_source(contract: StreamContract) -> bool:
    return contract.transport.endpoint is Endpoint.TARGET


def _is_top_sink(contract: StreamContract) -> bool:
    return contract.transport.endpoint is Endpoint.INITIATOR


class StreamLink(Space):
    """One declared connection between a producer port and a consumer port."""

    name = Param(str)
    spec = Param(STREAM_SPEC)
    source = Param(STREAM_CONTRACT)
    sink = Param(STREAM_CONTRACT)
    source_instance = Param(str)
    sink_instance = Param(str)

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()

    @constraint
    def compatible(self) -> bool | Rejected:
        source, sink, stage = self.source, self.sink, self.stage()
        if stage.requirements is None:
            found = list(
                compatibility(
                    source,
                    sink,
                    source_is_top=_is_top_source(source),
                    sink_is_top=_is_top_sink(sink),
                )
            )
        else:
            assert stage.input is not None and stage.output is not None
            found = [
                *compatibility(
                    source, stage.input, source_is_top=_is_top_source(source), sink_is_top=False
                ),
                *compatibility(
                    stage.output, sink, source_is_top=False, sink_is_top=_is_top_sink(sink)
                ),
            ]
        return _refusal(self.name, found)

    @derived(semantics=CONNECTION)
    def link(self) -> Connection:
        source, sink = self.source, self.sink
        return Connection(
            self.name,
            None if _is_top_source(source) else self.source_instance,
            source,
            None if _is_top_sink(sink) else self.sink_instance,
            sink,
            self.stage(),
        )

    connection = View(link, constraints=(compatible,))


def _refusal(stream: str, found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(
        found[0].code,
        "; ".join(f"{item.code}: {item.message}" for item in found),
        values={"stream": stream},
    )


class BufferedStreamLink(StreamLink):
    transport = SubspaceChoice(
        {"direct": Subspace(_Direct), "fifo": Subspace(StreamFifo, spec=StreamLink.spec)},
        exports=(STAGE,),
    )
    stage = View(transport.accepted(STAGE))


End = tuple[str, ValueRef[StreamContract]]


class Stream(Subspace[StreamLink]):
    """Declare a connection from ``source`` to ``sink``, each (instance name, port view).

    Instance names are the physical names of the endpoints' modules; a boundary
    endpoint's name is unused. The declared attribute name names the stream
    and its FIFO instance.
    """

    def __init__(
        self, spec: ValueRef[StreamSpec], *, source: End, sink: End, buffered: bool = False
    ) -> None:
        super().__init__(
            BufferedStreamLink if buffered else StreamLink,
            spec=spec,
            source=source[1],
            sink=sink[1],
            source_instance=source[0],
            sink_instance=sink[0],
        )
        self.buffered = buffered

    def __set_name__(self, owner: type[object], name: str) -> None:
        super().__set_name__(owner, name)
        self.bindings = MappingProxyType({**self.bindings, "name": name})

    @property
    def spec(self) -> ValueRef[StreamSpec]:
        return self.ref(StreamLink.spec)

    @property
    def connection(self) -> ValueRef[Connection]:
        return self.accepted(StreamLink.connection)


def connected(stream: Stream) -> Constraint:
    """A parent constraint that holds exactly when ``stream``'s connection is accepted."""

    @constraint(connection=stream.connection)
    def accepted(*, connection: Connection) -> bool:
        return True

    return accepted


@dataclass(frozen=True)
class Composed:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements


COMPOSED = default_semantics(Composed)


def compose(
    *,
    module: str,
    producer: ProducerIdentity,
    instances: Mapping[str, ModuleBuildRequirements | None],
    connections: Sequence[Connection],
) -> Composed:
    """Wire accepted instances through accepted connections; a pure reduction.

    ``instances`` maps instance names to modules (None omits a boundary case).
    Raises ``StreamMismatch`` or ``PhysicalStructureError`` only for a defect the
    streams' own constraints cannot see, such as a clock-domain conflict.
    """
    placed = {name: module for name, module in instances.items() if module is not None}
    fifos = {c.name: c.stage for c in connections if c.stage.requirements is not None}
    boundary = [c.source for c in connections if c.source_owner is None] + [
        c.sink for c in connections if c.sink_owner is None
    ]
    children = [*placed.values(), *(s.requirements for s in fifos.values() if s.requirements)]
    composition = Composition(_top_abi(module, boundary, children))
    stream_pins: dict[str, set[str]] = {}
    for c in connections:
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
    for c in connections:
        source, sink = StreamEnd(c.source_owner, c.source), StreamEnd(c.sink_owner, c.sink)
        if c.stage.input is not None and c.stage.output is not None:
            instance = f"u_{c.name}_fifo"
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
    "COMPOSED",
    "CONNECTION",
    "Composed",
    "Connection",
    "MODULE",
    "Module",
    "OUTPUT_PORT",
    "Port",
    "STREAM_SPEC",
    "Stage",
    "Stream",
    "StreamFifo",
    "StreamLink",
    "StreamSpec",
    "TopInput",
    "TopOutput",
    "compose",
    "connected",
]
