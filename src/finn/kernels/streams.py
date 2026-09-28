# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces that kernels reference.

A ``Stream`` is the physical form of the logical stream
(``finn.dataflow.stream``): a node of its own, declared in the composite
beside the kernels it joins, carrying one ``tensor`` supplied by the
composite. A kernel has one reference input per stream it sits on
(``output_stream: Stream = Param()``) and exports, under ``PORT``, one
contract per input: ``exports = {PORT: {output_stream: output_port}}``. Each
end presents its own traversal of the tensor in that contract, reading only
the stream's ``tensor``. The stream sees the kernels that reference it through
``users = Users(PORT)``, each with only the port it presents on this stream,
so a port's refusal stays on its own stream. The contract's transport
endpoint says whether the kernel produces into the stream (initiator) or
consumes from it (target). The one-producer-one-consumer rule, compatibility
and the AXIS boundary belong to this family, not to the engine; the tensor
each end must traverse and the plan are the logical stream's.

A stream with a user on one side only is a boundary of its composite. Its
``port`` input names the top-level AXIS port (``in0_V``): an ABI name is
design data of the stream, independent of the stream's node name, which is
its identity and the prefix of its persisted decision keys. The boundary
presents what its internal end presents, by one rule: an input boundary
without the replay its receiver realizes (``unreplayed``), an output boundary
as produced, neither with markers and both as a single pass.

The logical stream derives the ``plan`` between the two ends' beat
sequences and refuses one it cannot carry out (``stream-plan``); ``ends`` is
the contracts' logical part. A non-empty plan opens the stream's ``adapter``
Decision over nodes, whose candidates are fixed chains of FinnLib modules
(``finn.kernels.adapters``); each refuses a plan it does not carry out, so at
most one survives. A stream whose ``adaptable`` input is False admits no
adapter and refuses any plan. The adapter's modules are the stream's stages,
each checked on both of its sides.

A ``BufferedStream`` owns a ``transport`` Decision over two nodes, ``direct``
and ``fifo``, after its adapter. The FIFO candidate owns its ``depth`` and the
FIFO's ``ram_style``; it is an identity stage presenting what arrives at it.
``Members(CONNECTION)`` collects a composite's streams and ``netlist`` wires
them, each stage as ``u_<stream>_<stage>``; instance names come from the
located user names, never from literals.

A composite is one generated module with one clock: ``ap_clk`` and the
active-low reset ``ap_rst_n``, plus ``ap_clk2x`` when a child needs an aligned
double-rate clock. ``netlist`` drives each child clock and reset pin from its
declared role, never from its name; wiring the module's instance into a
design is a consumer of that interface, not the kernel's. It also takes
``Members(TIEOFFS)`` (inputs a kernel holds constant and outputs it leaves
unconnected in its configuration) and ``Members(EXPORTED)`` (control buses
presented at the module boundary).
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
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Endpoint,
    Free,
    Reset,
    Signal,
)
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
from finn.kernels.base import MODULE, TIEOFFS, TIEOFFS_SEMANTICS, Tieoffs
from finn.kernels.control import Exported, top_bus
from finn.kernels.fifo import FifoKernel
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import (
    STREAM_CONTRACT,
    Mismatch,
    StreamContract,
    compatibility,
)
from finn.dataflow.stream import ENDS, End, Ends
from finn.dataflow.stream import Stream as LogicalStream
from finn.dataflow.tensor import TENSOR, ScalarEncoding, Tensor
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence, unreplayed
from finn.kernels.adapters import (
    INPUT_GEN_RAM_STYLES,
    STAGE_SEMANTICS,
    STAGES,
    InputGenAdapter,
    RegroupAdapter,
    RegroupMarkersAdapter,
    ReorderWidthAdapter,
    ReorderWidthMarkersAdapter,
    Stage,
    StreamAdapter,
    WidthAdapter,
    WidthReorderAdapter,
    buffers,
)
from finn.kernels.physical.lowering import lower_module_structure
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.physical.validation import abi_pins


# The composed module's clocking pins: its interface convention, not a routing rule.
CLOCK, CLOCK2X, RESET = "ap_clk", "ap_clk2x", "ap_rst_n"


def boundary_contract(
    name: str, element: ScalarEncoding, sequence: BeatSequence, endpoint: Endpoint
) -> StreamContract:
    """The AXIS port a composed module presents for one of its own streams."""
    if len(sequence.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    form = sequence.form
    stream = AxiStream(
        name, element.dtype, form.lanes, endpoint=endpoint, last=bool(sequence.markers)
    )
    transport = stream.native(clock=CLOCK, reset=RESET)
    markers = {transport.markers[0].signal: sequence.markers[0]} if sequence.markers else {}
    return StreamContract(transport, element, form, sequence.repetition, markers)


def boundary_sequence(internal: StreamContract, *, receiving: bool) -> BeatSequence:
    """What a boundary presents for its internal end: the receiver realizes replay.

    An input boundary (``receiving``) presents its consumer's form without
    replay; an output boundary presents its producer's form. Neither carries
    markers, and a boundary streams a single pass.
    """
    form = unreplayed(internal.form) if receiving else internal.form
    return BeatSequence(form)


class _Direct(Space):
    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()


class StreamFifo(Space):
    """An identity adapter: what arrives at it, presented on both sides of a native FIFO."""

    tensor: Tensor = Param(semantics=TENSOR)
    arriving: BeatSequence = Param(semantics=BEAT_SEQUENCE)

    @derived
    def word_bits(self) -> int:
        return self.arriving.form.lanes * self.tensor.element.bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        element, arriving = self.tensor.element, self.arriving
        source, sink = self.buffer.interfaces
        return Stage(
            self.buffer.build_requirements,
            StreamContract(source, element, arriving.form, arriving.repetition),
            StreamContract(sink, element, arriving.form, arriving.repetition),
            "fifo",
        )


PORT = ViewKey("port", STREAM_CONTRACT)
"""A kernel's port on one stream, exported per reference input."""


@dataclass(frozen=True)
class Connection:
    """One checked stream; an owner of None is the composed module itself."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract
    stages: tuple[Stage, ...] = ()


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


class Stream(LogicalStream):
    """A relation between the kernels that reference it: one producer, one consumer.

    ``users`` holds the port each present user presents on this stream,
    located by the user's name and the input it references this stream
    through. A side without a user is the composite's boundary, presented as
    the AXIS port ``port`` (``boundary_sequence``).
    """

    port: str = Param(required=False)
    users = Users(PORT)

    @derived(semantics=ENDPOINTS)
    def endpoints(self) -> Endpoints | Rejected:
        producers: list[tuple[str | None, StreamContract]] = []
        consumers: list[tuple[str | None, StreamContract]] = []
        for end in self.users:
            contract = end.value
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((end.node, contract))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(f"{end.node}.{end.member}" for end in self.users)
            return reject(
                "stream-users",
                f"a stream has at most one producer and one consumer; referenced by {named}",
            )
        if not producers and not consumers:
            return reject("stream-unused", "no present kernel references this stream")
        try:
            # A side without a user is the boundary. Seen from inside, the
            # composite's input is the AXIS target and its output the initiator.
            if not producers:
                inside = consumers[0][1]
                producers.append((None, self._boundary(inside, Endpoint.TARGET)))
            if not consumers:
                inside = producers[0][1]
                consumers.append((None, self._boundary(inside, Endpoint.INITIATOR)))
        except ValueError as error:
            return reject("stream-boundary", str(error))
        (source_owner, source), (sink_owner, sink) = producers[0], consumers[0]
        return Endpoints(source_owner, source, sink_owner, sink)

    def _boundary(self, inside: StreamContract, endpoint: Endpoint) -> StreamContract:
        receiving = endpoint is Endpoint.TARGET
        sequence = boundary_sequence(inside, receiving=receiving)
        return boundary_contract(self.port, self.tensor.element, sequence, endpoint)

    @derived(semantics=ENDS)
    def ends(self) -> Ends:
        """The logical part of the two contracts: element and beat sequence."""
        found = self.endpoints
        return Ends(
            End(found.source_owner, found.source.element, found.source.sequence),
            End(found.sink_owner, found.sink.element, found.sink.sequence),
        )

    @derived
    def buffering(self) -> bool:
        """Whether the adapter the plan takes has an ``input_gen``, whose memory is a choice."""
        return self.adapting and buffers(self.plan)

    adapter_ram_style: str = Decision(values=INPUT_GEN_RAM_STYLES, when=buffering)
    adapter: StreamAdapter = Decision(
        values={
            "input_gen": InputGenAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
            "vpc": WidthAdapter(tensor=LogicalStream.tensor, plan=LogicalStream.plan),
            "vpc_input_gen": WidthReorderAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
            "input_gen_vpc": ReorderWidthAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
            "input_gen_vpc_input_gen": ReorderWidthMarkersAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
            "vpc_input_gen_vpc": RegroupAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
            "vpc_input_gen_vpc_input_gen": RegroupMarkersAdapter(
                tensor=LogicalStream.tensor, plan=LogicalStream.plan, ram_style=adapter_ram_style
            ),
        },
        when=LogicalStream.adapting,
    )
    adapter_admitted = View(adapter.admitted)
    adapter_stages = View(adapter.stages)

    @derived(semantics=STAGES)
    def adapted(self) -> tuple[Stage, ...]:
        """The adapter's stages, in order; none when the ends connect directly."""
        return self.adapter_stages if self.adapting else ()

    @derived(semantics=BEAT_SEQUENCE)
    def arriving(self) -> BeatSequence:
        """What arrives after the adapter: the beat sequence a transport stage receives."""
        adapted = self.adapted
        output = adapted[-1].output if adapted else None
        return self.endpoints.source.sequence if output is None else output.sequence

    @view(semantics=STAGES)
    def stages(self) -> tuple[Stage, ...]:
        return self.adapted

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop, source through every stage to sink, connects directly."""
        ends = self.endpoints
        found: list[Mismatch] = []
        current, current_top = ends.source, ends.source_owner is None
        for stage in self.stages:
            assert stage.input is not None and stage.output is not None
            found += compatibility(
                current, stage.input, source_is_top=current_top, sink_is_top=False
            )
            current, current_top = stage.output, False
        found += compatibility(
            current, ends.sink, source_is_top=current_top, sink_is_top=ends.sink_owner is None
        )
        return _refusal(found)

    @derived(semantics=CONNECTION_SEMANTICS)
    def link(self) -> Connection:
        ends = self.endpoints
        return Connection(ends.source_owner, ends.source, ends.sink_owner, ends.sink, self.stages)

    connection = View(
        link, requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible)
    )
    exports = {CONNECTION: connection}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


class BufferedStream(Stream):
    """A stream whose ``transport`` after its adapter is direct or a FIFO."""

    transport: _Direct | StreamFifo = Decision(
        values={
            "direct": _Direct(),
            "fifo": StreamFifo(tensor=Stream.tensor, arriving=Stream.arriving),
        }
    )
    transport_stage = View(transport.stage)

    @view(semantics=STAGES)
    def stages(self) -> tuple[Stage, ...]:
        fifo = self.transport_stage
        return (*self.adapted, *((fifo,) if fifo.requirements is not None else ()))


@dataclass(frozen=True)
class Composed:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements


COMPOSED = default_semantics(Composed)


def _instance(node: str | None) -> str | None:
    """``u_<node>``; a candidate of a Decision (``memory.rom``) joins with ``_``."""
    return None if node is None else "u_" + node.replace(".", "_")


def _owner(node: str | None, modules: set[str]) -> str | None:
    """The module a stream end belongs to: its nearest ancestor that owns a module.

    A kernel's ``Port`` node presents the end (``compute.packed.x``); the
    instance is its kernel's (``compute.packed``). A node owning no module and
    below none is its own owner.
    """
    if node is None:
        return None
    parts = node.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in modules:
            return candidate
    return node


def netlist(
    modules: Sequence[Located[ModuleBuildRequirements]],
    streams: Sequence[Located[Connection]],
    tieoffs: Sequence[Located[Tieoffs]] = (),
    controls: Sequence[Located[tuple[Exported, ...]]] = (),
    *,
    module: str,
    producer: ProducerIdentity,
) -> Composed | Rejected:
    """Wire the composite's modules through its streams and control buses.

    Each module is instantiated as ``u_<node>``, each stage of a stream as
    ``u_<stream>_<stage>`` (``u_activations_input_gen``, ``u_weights_fifo``). A
    stream end presented by a kernel's port belongs to that kernel's instance.
    Every clock and reset pin is driven by its role (a free clock from
    ``ap_clk``, a clock at twice another from ``ap_clk2x``, a reset from
    ``ap_rst_n``), every exported control bus is wired through to its top port,
    and every tied input is held by its kernel's ``Tieoffs``. An input nothing
    drives is refused, as is a defect no single stream can see, such as a
    clock-domain conflict.
    """
    owners = {str(item.node) for item in modules}
    connections = [
        (
            str(stream.node).replace(".", "_"),
            replace(
                stream.value,
                source_owner=_instance(_owner(stream.value.source_owner, owners)),
                sink_owner=_instance(_owner(stream.value.sink_owner, owners)),
            ),
        )
        for stream in streams
    ]
    try:
        return _wire(
            {str(_instance(item.node)): item.value for item in modules},
            connections,
            {str(_instance(item.node)): item.value for item in tieoffs},
            [exported for item in controls for exported in item.value],
            module,
            producer,
        )
    except ValueError as error:
        return reject("stream-composition", str(error))


def _wire(
    placed: dict[str, ModuleBuildRequirements],
    connections: list[tuple[str, Connection]],
    tieoffs: dict[str, Tieoffs],
    exported: list[Exported],
    module: str,
    producer: ProducerIdentity,
) -> Composed:
    staged = [
        (f"u_{name}_{stage.name}", stage)
        for name, c in connections
        for stage in c.stages
        if stage.requirements is not None
    ]
    boundary = [c.source for _, c in connections if c.source_owner is None] + [
        c.sink for _, c in connections if c.sink_owner is None
    ]
    children = [
        *placed.values(),
        *(stage.requirements for _, stage in staged if stage.requirements),
    ]
    doubled = any(_clock_roles(child.abi).get(CLOCK2X) for child in children)
    tops = [top_bus(item.child, item.port, CLOCK, RESET) for item in exported]
    composition = Composition(_top_abi(module, boundary, doubled, tops))
    stream_pins: dict[str, set[str]] = {}
    for _, c in connections:
        for owner, contract in ((c.source_owner, c.source), (c.sink_owner, c.sink)):
            if owner is not None:
                stream_pins.setdefault(owner, set()).update(
                    pin.name for pin in contract.transport.pins()
                )
    for item in exported:
        stream_pins.setdefault(str(_instance(item.node)), set()).update(
            member.physical for member in item.child.signals
        )
    for name, requirements in placed.items():
        composition.add(name, requirements)
        _drive(
            composition,
            name,
            requirements,
            stream_pins.get(name, set()),
            tieoffs.get(name, Tieoffs()),
        )
    for item in exported:
        composition.export(
            str(_instance(item.node)), item.child, top_bus(item.child, item.port, CLOCK, RESET)
        )
    for instance, stage in staged:
        assert stage.requirements and stage.input and stage.output
        composition.add(instance, stage.requirements)
        pins = {p.name for p in (*stage.input.transport.pins(), *stage.output.transport.pins())}
        _drive(composition, instance, stage.requirements, pins, Tieoffs())
    for name, c in connections:
        source = StreamEnd(c.source_owner, c.source)
        for stage in c.stages:
            if stage.input is None or stage.output is None:
                continue
            instance = f"u_{name}_{stage.name}"
            composition.connect(source, StreamEnd(instance, stage.input))
            source = StreamEnd(instance, stage.output)
        composition.connect(source, StreamEnd(c.sink_owner, c.sink))
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


def _clock_roles(abi: ModuleABIRequirements) -> dict[str, list[str]]:
    """A child's clock and reset pins by the top pin their role binds them to."""
    roles: dict[str, list[str]] = {}
    for name, info in abi_pins(abi).items():
        if info.bus_id is not None or info.direction is not Direction.IN:
            continue
        if isinstance(info.role, Clock):
            rate = info.role.rate
            if isinstance(rate, Derived):
                if rate.ratio != 2:
                    raise ValueError(f"{name}: only a clock at twice ap_clk can be supplied")
                roles.setdefault(CLOCK2X, []).append(name)
            else:
                roles.setdefault(CLOCK, []).append(name)
        elif isinstance(info.role, Reset):
            roles.setdefault(RESET, []).append(name)
    return roles


def _top_abi(
    module: str, boundary: list[StreamContract], doubled: bool, controls: list[Bus]
) -> ModuleABIRequirements:
    """Clocks and reset, the boundary streams, then the control buses."""
    clocks = (CLOCK, CLOCK2X) if doubled else (CLOCK,)
    return ModuleABIRequirements(
        GeneratedModuleName(module),
        (
            Signal(CLOCK, Direction.IN, 1, Clock(Free())),
            *((Signal(CLOCK2X, Direction.IN, 1, Clock(Derived(CLOCK, 2))),) if doubled else ()),
            Signal(
                RESET,
                Direction.IN,
                1,
                Reset(active_low=True, synchronous=True, synchronous_to=clocks),
            ),
            *(contract.transport.axis_bus() for contract in boundary),
            *controls,
        ),
        (),
        (ClockAlignment(CLOCK, CLOCK2X),) if doubled else (),
    )


def _drive(
    composition: Composition,
    instance: str,
    requirements: ModuleBuildRequirements,
    stream_pins: set[str],
    tieoffs: Tieoffs,
) -> None:
    """Drive every input outside the instance's streams by its role or its tie-off."""
    clocking = {pin: top for top, pins in _clock_roles(requirements.abi).items() for pin in pins}
    tied = dict(tieoffs.inputs)
    for name, info in abi_pins(requirements.abi).items():
        if info.direction is not Direction.IN or name in stream_pins:
            continue
        if name in tied:
            composition.tie(instance, name, tied[name])
        elif name in clocking:
            composition.drive(instance, name, clocking[name])
        elif info.bus_id is None:
            raise ValueError(f"{instance}.{name}: an input outside every stream has no driver")
    for name in tieoffs.unused:
        composition.dispose(instance, name, "tied off")


__all__ = [
    "BufferedStream",
    "CLOCK",
    "CLOCK2X",
    "COMPOSED",
    "CONNECTION",
    "Composed",
    "Connection",
    "Endpoints",
    "MODULE",
    "PORT",
    "RESET",
    "Stage",
    "Stream",
    "StreamFifo",
    "TIEOFFS",
    "TIEOFFS_SEMANTICS",
    "Tieoffs",
    "boundary_contract",
    "boundary_sequence",
    "netlist",
]
