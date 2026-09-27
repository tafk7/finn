# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces that kernels reference.

A ``Stream`` is a node of its own, declared in the composite beside the
kernels it joins. A kernel has one reference input per stream it sits on
(``output_stream: Stream = Param()``) and exports, under ``PORT``, one contract
per input: ``exports = {PORT: {output_stream: output_port}}``. The stream sees
the kernels that reference it through ``Users(PORT)``, each with only the port
it presents on this stream, so a port's refusal stays on its own stream. The
contract's transport endpoint says whether the kernel produces into the stream
(initiator) or consumes from it (target). The one-producer-one-consumer rule,
compatibility and the AXIS boundary belong to this family, not to the engine.

A stream with a user on one side only is a boundary of its composite. Its
``port`` input names the top-level AXIS port (``in0_V``): an ABI name is
design data of the stream, independent of the stream's node name, which is
its identity and the prefix of its persisted decision keys.

A stream's ``spec`` is supplied by the composite and must not depend on its
users: kernels read it (``self.output_stream.spec``) to build their port
contracts, so a spec derived from a user's contract is a dependency cycle.

A ``BufferedStream`` owns a ``transport`` Decision over two nodes, ``direct``
and ``fifo``. The FIFO candidate owns its ``depth`` and the FIFO's
``ram_style``; it is an identity adapter, checked on both of its sides.
``Members(CONNECTION)`` collects a composite's streams and ``netlist`` wires
them; instance names come from the located user names, never from literals.

A stream lives in a clock domain (``clock``, a ``ClockDomain`` node): its AXIS
boundary is associated with the domain's pins, and a transport stage runs in
it. ``netlist`` also takes the composite's ``Members(DOMAIN)``, which drive
every clock and reset pin, and ``Members(TIEOFFS)``: inputs a kernel holds
constant and outputs it leaves unconnected in its configuration. No pin is
routed by its name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finn.core.space import (
    Decision,
    Located,
    Param,
    Rejected,
    Space,
    Users,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    domain,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
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
from finn.kernels.clocks import ClockDomain, Domain
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


MODULE = ViewKey("module", default_semantics(ModuleBuildRequirements))


@dataclass(frozen=True)
class Tieoffs:
    """Pins a kernel leaves out of the composition in this configuration.

    ``inputs`` are held constant, as (pin, value); ``unused`` outputs are left
    unconnected.
    """

    inputs: tuple[tuple[str, int], ...] = ()
    unused: tuple[str, ...] = ()


TIEOFFS_SEMANTICS = default_semantics(Tieoffs)
TIEOFFS = ViewKey("tieoffs", TIEOFFS_SEMANTICS)


def boundary_contract(
    name: str, spec: StreamSpec, endpoint: Endpoint, *, clock: str, reset: str | None
) -> StreamContract:
    """The AXIS port a composed module presents for one of its own streams."""
    if len(spec.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    stream = AxiStream(
        name, spec.element.dtype, spec.form.lanes, endpoint=endpoint, last=bool(spec.markers)
    )
    transport = stream.native(clock=clock, reset=reset)
    markers = {transport.markers[0].signal: spec.markers[0]} if spec.markers else {}
    return StreamContract(transport, spec.element, spec.form, spec.repetition, markers)


@dataclass(frozen=True)
class Stage:
    """A transport stage inside a stream: none (direct) or a module with two ports."""

    requirements: ModuleBuildRequirements | None = None
    input: StreamContract | None = None
    output: StreamContract | None = None


STAGE_SEMANTICS = default_semantics(Stage)


class _Direct(Space):
    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()


class StreamFifo(Space):
    """An identity adapter: the stream's own spec on both sides of a native FIFO."""

    spec: StreamSpec = Param(semantics=STREAM_SPEC)

    @derived
    def word_bits(self) -> int:
        return self.spec.payload_bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        spec = self.spec
        source, sink = self.buffer.interfaces
        return Stage(
            self.buffer.build_requirements,
            StreamContract(source, spec.element, spec.form),
            StreamContract(sink, spec.element, spec.form),
        )


PORT = ViewKey("port", STREAM_CONTRACT)
"""A kernel's port on one stream, exported per reference input."""


@dataclass(frozen=True)
class Connection:
    """One checked stream; an owner of None is the composed module itself.

    A stage runs in the stream's clock domain: ``clocking`` holds that
    domain's top clock and reset pins when there is a stage.
    """

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract
    stage: Stage
    clocking: tuple[str, str | None] | None = None


CONNECTION_SEMANTICS = default_semantics(Connection)
CONNECTION = ViewKey("connection", CONNECTION_SEMANTICS)


@dataclass(frozen=True)
class Endpoints:
    """The producing and consuming ends of one stream; an owner of None is the boundary."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract


ENDPOINTS = default_semantics(Endpoints)


class Stream(Space):
    """A relation between the kernels that reference it: one producer, one consumer.

    ``ends`` holds the port each present user presents on this stream, located
    by the user's name and the input it references this stream through. A side
    without a user is the composite's boundary, presented as the AXIS port
    ``port`` in the stream's ``clock`` domain, which also clocks a transport
    stage. A stream with neither needs no domain.
    """

    spec: StreamSpec = Param(semantics=STREAM_SPEC)
    port: str = Param(required=False)
    clock: ClockDomain = Param(required=False)
    ends = Users(PORT)

    @derived(semantics=ENDPOINTS)
    def endpoints(self) -> Endpoints | Rejected:
        producers: list[tuple[str | None, StreamContract]] = []
        consumers: list[tuple[str | None, StreamContract]] = []
        for end in self.ends:
            contract = end.value
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((end.node, contract))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(f"{end.node}.{end.member}" for end in self.ends)
            return reject(
                "stream-users",
                f"a stream has at most one producer and one consumer; referenced by {named}",
            )
        if not producers and not consumers:
            return reject("stream-unused", "no present kernel references this stream")
        try:
            # A side without a user is the boundary. Seen from inside, the
            # composite's input is the AXIS target and its output the initiator.
            source = producers[0] if producers else (None, self._boundary(Endpoint.TARGET))
            sink = consumers[0] if consumers else (None, self._boundary(Endpoint.INITIATOR))
        except ValueError as error:
            return reject("stream-boundary", str(error))
        return Endpoints(source[0], source[1], sink[0], sink[1])

    def _boundary(self, endpoint: Endpoint) -> StreamContract:
        domain = self.clock
        return boundary_contract(
            self.port, self.spec, endpoint, clock=domain.clock, reset=domain.reset
        )

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()

    @constraint
    def compatible(self) -> bool | Rejected:
        ends = self.endpoints
        stage = self.stage
        source_top, sink_top = ends.source_owner is None, ends.sink_owner is None
        if stage.requirements is None:
            found = list(
                compatibility(
                    ends.source, ends.sink, source_is_top=source_top, sink_is_top=sink_top
                )
            )
        else:
            assert stage.input is not None and stage.output is not None
            found = [
                *compatibility(
                    ends.source, stage.input, source_is_top=source_top, sink_is_top=False
                ),
                *compatibility(stage.output, ends.sink, source_is_top=False, sink_is_top=sink_top),
            ]
        return _refusal(found)

    @derived(semantics=CONNECTION_SEMANTICS)
    def link(self) -> Connection:
        ends, stage = self.endpoints, self.stage
        clocking = None
        if stage.requirements is not None:
            clocking = (self.clock.clock, self.clock.reset)
        return Connection(
            ends.source_owner, ends.source, ends.sink_owner, ends.sink, stage, clocking
        )

    connection = View(link, requires=(compatible,))
    exports = {CONNECTION: connection}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


class BufferedStream(Stream):
    transport: _Direct | StreamFifo = Decision(
        values={"direct": _Direct(), "fifo": StreamFifo(spec=Stream.spec)}
    )
    stage = View(transport.stage)


@dataclass(frozen=True)
class Composed:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements


COMPOSED = default_semantics(Composed)


def _instance(node: str | None) -> str | None:
    """``u_<node>``; a candidate of a Decision (``implementation.cyclic``) joins with ``_``."""
    return None if node is None else "u_" + node.replace(".", "_")


def netlist(
    modules: Sequence[Located[ModuleBuildRequirements]],
    streams: Sequence[Located[Connection]],
    domains: Sequence[Located[Domain]],
    tieoffs: Sequence[Located[Tieoffs]] = (),
    *,
    module: str,
    producer: ProducerIdentity,
) -> Composed | Rejected:
    """Wire the composite's modules through its streams and clock domains.

    Each module is instantiated as ``u_<node>``, a FIFO stage as ``u_<stream>_fifo``.
    Every clock and reset pin is driven from the domain that names it and every
    tied input from its kernel's ``Tieoffs``; an input nothing drives is refused, as
    is a defect no single stream can see, such as a clock-domain conflict.
    """
    connections = [
        (
            str(stream.node).replace(".", "_"),
            replace(
                stream.value,
                source_owner=_instance(stream.value.source_owner),
                sink_owner=_instance(stream.value.sink_owner),
            ),
        )
        for stream in streams
    ]
    try:
        return _wire(
            {str(_instance(item.node)): item.value for item in modules},
            connections,
            [item.value for item in domains],
            {str(_instance(item.node)): item.value for item in tieoffs},
            module,
            producer,
        )
    except ValueError as error:
        return reject("stream-composition", str(error))


def _wire(
    placed: dict[str, ModuleBuildRequirements],
    connections: list[tuple[str, Connection]],
    domains: list[Domain],
    tieoffs: dict[str, Tieoffs],
    module: str,
    producer: ProducerIdentity,
) -> Composed:
    fifos = {n: c for n, c in connections if c.stage.requirements is not None}
    boundary = [c.source for _, c in connections if c.source_owner is None] + [
        c.sink for _, c in connections if c.sink_owner is None
    ]
    composition = Composition(_top_abi(module, boundary, domains))
    clocking = _clocking(domains)
    stream_pins: dict[str, set[str]] = {}
    for _, c in connections:
        for owner, contract in ((c.source_owner, c.source), (c.sink_owner, c.sink)):
            if owner is not None:
                stream_pins.setdefault(owner, set()).update(
                    pin.name for pin in contract.transport.pins()
                )
    for name, requirements in placed.items():
        composition.add(name, requirements)
        _drive(
            composition,
            name,
            requirements,
            stream_pins.get(name, set()),
            clocking.get(name, {}),
            tieoffs.get(name, Tieoffs()),
        )
    for stream, c in fifos.items():
        stage = c.stage
        assert stage.requirements and stage.input and stage.output and c.clocking
        instance = f"u_{stream}_fifo"
        composition.add(instance, stage.requirements)
        pins = {p.name for p in (*stage.input.transport.pins(), *stage.output.transport.pins())}
        _drive(
            composition, instance, stage.requirements, pins, _stage_clocking(stage, c), Tieoffs()
        )
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


def _clocking(domains: list[Domain]) -> dict[str, dict[str, str]]:
    """Instance -> child clock or reset pin -> the top pin that drives it."""
    driven: dict[str, dict[str, str]] = {}
    for item in domains:
        if item.clock is None:
            continue
        for node, pins in item.driven:
            instance = driven.setdefault(str(_instance(node)), {})
            if pins.clock is not None:
                instance[pins.clock] = item.clock.name
            if pins.reset is not None:
                assert item.reset is not None  # a derived domain refuses reset pins
                instance[pins.reset] = item.reset
    return driven


def _stage_clocking(stage: Stage, connection: Connection) -> dict[str, str]:
    """A stage runs in its stream's domain: its clock and reset pins, by role."""
    assert stage.requirements is not None and connection.clocking is not None
    clock, reset = connection.clocking
    pins: dict[str, str] = {}
    for name, info in abi_pins(stage.requirements.abi).items():
        if isinstance(info.role, Clock):
            pins[name] = clock
        elif isinstance(info.role, Reset) and reset is not None:
            pins[name] = reset
    return pins


def _top_abi(
    module: str, boundary: list[StreamContract], domains: list[Domain]
) -> ModuleABIRequirements:
    """The used domains' clocks, then their resets, then the boundary buses."""
    used = [item for item in domains if item.clock is not None]
    clocks = [item.clock for item in used if item.clock is not None]
    names = {clock.name for clock in clocks}
    missing = [item.base for item in used if item.base and item.base not in names]
    if missing:
        raise ValueError(f"a derived clock runs from {missing[0]}, which no kernel uses")
    resets = [
        Signal(
            item.reset,
            Direction.IN,
            1,
            Reset(
                active_low=True,
                synchronous=True,
                synchronous_to=(
                    item.clock.name,
                    *(
                        derived.clock.name
                        for derived in used
                        if derived.clock is not None and derived.base == item.clock.name
                    ),
                ),
            ),
        )
        for item in used
        if item.clock is not None and item.reset is not None
    ]
    return ModuleABIRequirements(
        GeneratedModuleName(module),
        (*clocks, *resets, *(contract.transport.axis_bus() for contract in boundary)),
        (),
        tuple(item.alignment for item in used if item.alignment is not None),
    )


def _drive(
    composition: Composition,
    instance: str,
    requirements: ModuleBuildRequirements,
    stream_pins: set[str],
    clocking: dict[str, str],
    tieoffs: Tieoffs,
) -> None:
    """Drive every input outside the instance's streams from its domain or its tie-off."""
    tied = dict(tieoffs.inputs)
    for name, info in abi_pins(requirements.abi).items():
        if info.direction is not Direction.IN or name in stream_pins:
            continue
        if name in clocking:
            composition.drive(instance, name, clocking[name])
        elif name in tied:
            composition.tie(instance, name, tied[name])
        elif info.bus_id is None:
            raise ValueError(f"{instance}.{name}: an input outside every stream has no driver")
    for name in tieoffs.unused:
        composition.dispose(instance, name, "tied off")


__all__ = [
    "BufferedStream",
    "COMPOSED",
    "CONNECTION",
    "Composed",
    "Connection",
    "Endpoints",
    "MODULE",
    "PORT",
    "STREAM_SPEC",
    "Stage",
    "Stream",
    "StreamFifo",
    "StreamSpec",
    "TIEOFFS",
    "TIEOFFS_SEMANTICS",
    "Tieoffs",
    "boundary_contract",
    "netlist",
]
