# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declared streams: kernels bind their ports to named connections in a parent Space.

A parent declares each connection as a ``Stream`` over a ``StreamSpec`` (the
logical sequence: element, traversal, repetition, markers) and binds child
``Port`` formals to ``stream.spec``. A placement may be a ``SubspaceChoice``,
so either end of a stream can be a choice of kernels; every case must bind the
same ports to the same streams. ``assemble_streams`` then derives the whole
physical structure: each stream's producer and consumer contracts are checked
and wired by ``Composition.connect``, clocks and resets are routed by role, and
boundary placements (``TopInput``/``TopOutput``) become top-level AXIS ports.

A stream declared ``buffered=True`` owns a transport choice between ``direct``
and ``fifo``. The FIFO case owns its ``depth`` and the FIFO's ``ram_style``
decisions; it is an identity adapter, so the stream's contract is unchanged.
Whether a FIFO is needed and how deep it must be is a compiler question; this
layer only makes the slot available where the kernel author allows one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from finn.core.space import (
    Decision,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    ValueRef,
    View,
    ViewKey,
    default_semantics,
    derived,
    domain,
    view,
)
from finn.core.space.declarations import Declaration, ScopedValueRef
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
from finn.kernels.physical.contract import StreamContract
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
    """A kernel's stream endpoint: bound by a parent to ``stream.spec``.

    Optional, so a kernel can be configured on its own without any stream.
    """

    def __init__(self, direction: Endpoint) -> None:
        super().__init__(STREAM_SPEC, required=False)
        self.direction = direction


@dataclass(frozen=True)
class Component:
    """What a placement contributes: its module (None for a boundary) and port contracts."""

    requirements: ModuleBuildRequirements | None
    ports: Mapping[str, StreamContract] = field(default_factory=dict)
    initializer: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", dict(self.ports))


COMPONENT_SEMANTICS = default_semantics(Component)
COMPONENT = ViewKey("component", COMPONENT_SEMANTICS)
_CLOCKING = {"clock": "ap_clk", "reset": "ap_rst_n"}


class TopInput(Space):
    """The composed module's own input port, producing a stream."""

    name = Param(str)
    output_stream = Port(Endpoint.INITIATOR)

    @view(semantics=COMPONENT_SEMANTICS)
    def component(self) -> Component:
        contract = _top(self.name, self.output_stream, Endpoint.TARGET)
        return Component(None, {"output_stream": contract})

    exports = {COMPONENT: component}


class TopOutput(Space):
    """The composed module's own output port, consuming a stream."""

    name = Param(str)
    input_stream = Port(Endpoint.TARGET)

    @view(semantics=COMPONENT_SEMANTICS)
    def component(self) -> Component:
        contract = _top(self.name, self.input_stream, Endpoint.INITIATOR)
        return Component(None, {"input_stream": contract})

    exports = {COMPONENT: component}


def _top(name: str, spec: StreamSpec, endpoint: Endpoint) -> StreamContract:
    if len(spec.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    stream = AxiStream(
        name, spec.element.dtype, spec.form.lanes, endpoint=endpoint, last=bool(spec.markers)
    )
    transport = stream.native(**_CLOCKING)
    markers = {transport.markers[0].signal: spec.markers[0]} if spec.markers else {}
    return StreamContract(transport, spec.element, spec.form, spec.repetition, markers)


class StreamLink(Space):
    """One declared connection; direct unless declared buffered."""

    spec = Param(STREAM_SPEC)

    @view(semantics=COMPONENT_SEMANTICS)
    def stage(self) -> Component:
        return Component(None)


class StreamFifo(Space):
    """An identity adapter: the stream's contract on both sides of a native FIFO."""

    spec = Param(STREAM_SPEC)

    @derived
    def word_bits(self) -> int:
        return self.spec.payload_bits

    buffer = Subspace(
        FifoKernel,
        word_bits=word_bits,
        depth=Decision(int, domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view(semantics=COMPONENT_SEMANTICS)
    def component(self) -> Component:
        spec = self.spec
        source, sink = self.buffer.interfaces()
        return Component(
            self.buffer.build_requirements(),
            {
                "input": StreamContract(source, spec.element, spec.form),
                "output": StreamContract(sink, spec.element, spec.form),
            },
        )

    exports = {COMPONENT: component}


class _Direct(Space):
    @view(semantics=COMPONENT_SEMANTICS)
    def component(self) -> Component:
        return Component(None)

    exports = {COMPONENT: component}


class BufferedStreamLink(StreamLink):
    transport = SubspaceChoice(
        {"direct": Subspace(_Direct), "fifo": Subspace(StreamFifo, spec=StreamLink.spec)},
        exports=(COMPONENT,),
    )
    stage = View(transport.accepted(COMPONENT))


class Stream(Subspace[StreamLink]):
    """Declare a connection; bind ports with ``Subspace(Kernel, port=stream.spec)``."""

    def __init__(self, spec: ValueRef[StreamSpec], *, buffered: bool = False) -> None:
        super().__init__(BufferedStreamLink if buffered else StreamLink, spec=spec)
        self.buffered = buffered

    @property
    def spec(self) -> ValueRef[StreamSpec]:
        return self.ref(StreamLink.spec)


@dataclass(frozen=True)
class _End:
    placement: str
    port: str
    direction: Endpoint


def _port_bindings(
    placement: Subspace[Space], streams: Mapping[int, str]
) -> dict[str, tuple[str, Endpoint]]:
    ports: dict[str, tuple[str, Endpoint]] = {}
    for formal, supplier in placement.bindings.items():
        declaration = getattr(placement.space_type, formal, None)
        if not isinstance(declaration, Port):
            continue
        if not isinstance(supplier, ScopedValueRef) or id(supplier.placement) not in streams:
            raise ValueError(f"port {formal!r} must be bound to a declared stream's spec")
        ports[formal] = (streams[id(supplier.placement)], declaration.direction)
    return ports


def _declarations(space_type: type[Space]) -> dict[str, Declaration]:
    found: dict[str, Declaration] = {}
    for cls in reversed(space_type.__mro__):
        for name, value in vars(cls).items():
            if isinstance(value, Declaration):
                found[name] = value
    return found


@dataclass(frozen=True)
class Assembled:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements
    components: Mapping[str, Component]


def assemble_streams(
    point: Space,
    *,
    module: str,
    producer: ProducerIdentity,
    instance_names: Mapping[str, str] | None = None,
) -> Assembled:
    """Wire every placement of ``point`` through its declared streams.

    Call from a view of the parent Space. Raises ``StreamMismatch`` for an
    incompatible pair and ``ValueError`` for a malformed stream topology.
    """
    names = dict(instance_names or {})
    declared = _declarations(type(point))
    streams = {id(value): name for name, value in declared.items() if isinstance(value, Stream)}
    ends: dict[str, list[_End]] = {name: [] for name in streams.values()}
    components: dict[str, Component] = {}
    for name, value in declared.items():
        if isinstance(value, Stream):
            continue
        if isinstance(value, SubspaceChoice):
            maps = [_port_bindings(case, streams) for case in value.alternatives.values()]
            if any(item != maps[0] for item in maps):
                raise ValueError(f"{name}: every case must bind the same ports to the same streams")
            bindings = maps[0]
            if not bindings:
                continue
            component = point.field(value.accepted(COMPONENT)).get()
        elif isinstance(value, Subspace):
            bindings = _port_bindings(value, streams)
            if not bindings:
                continue
            component = getattr(point, name).component()
        else:
            continue
        components[name] = component
        for port, (stream, direction) in bindings.items():
            ends[stream].append(_End(name, port, direction))

    instances: dict[str, str] = {}
    for name, component in components.items():
        if component.requirements is not None:
            instances[name] = names.get(name, "u_" + name)
    fifos: dict[str, Component] = {}
    for stream, link in ((s, getattr(point, s)) for s in ends):
        stage = link.stage()
        if stage.requirements is not None:
            fifos[stream] = stage

    boundary: list[StreamContract] = []
    for name, component in components.items():
        if component.requirements is None:
            boundary.extend(component.ports.values())
    children = [components[name].requirements for name in instances] + [
        stage.requirements for stage in fifos.values()
    ]
    top_abi = _top_abi(module, boundary, [item for item in children if item is not None])

    composition = Composition(top_abi)
    for name, instance in instances.items():
        requirements = components[name].requirements
        assert requirements is not None
        composition.add(instance, requirements)
        _drive(composition, instance, components[name])
    for stream, stage in fifos.items():
        assert stage.requirements is not None
        composition.add(f"u_{stream}_fifo", stage.requirements)
        _drive(composition, f"u_{stream}_fifo", stage)

    def end(item: _End) -> StreamEnd:
        return StreamEnd(instances.get(item.placement), components[item.placement].ports[item.port])

    for stream, items in ends.items():
        producers = [item for item in items if item.direction is Endpoint.INITIATOR]
        consumers = [item for item in items if item.direction is Endpoint.TARGET]
        if len(producers) != 1 or len(consumers) != 1:
            raise ValueError(
                f"stream {stream!r} needs one producer and one consumer "
                f"(found {len(producers)} and {len(consumers)}); fan-out is not supported"
            )
        source, sink = end(producers[0]), end(consumers[0])
        if stream in fifos:
            fifo = f"u_{stream}_fifo"
            composition.connect(source, StreamEnd(fifo, fifos[stream].ports["input"]))
            source = StreamEnd(fifo, fifos[stream].ports["output"])
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
    requirements = lower_module_structure(structure, producer=producer, wrapper_template=wrapper)
    return Assembled(structure, requirements, components)


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


def _drive(composition: Composition, instance: str, component: Component) -> None:
    """Route every input outside the component's streams: clocks and resets by role."""
    assert component.requirements is not None
    streams = {
        pin.name for contract in component.ports.values() for pin in contract.transport.pins()
    }
    for name, info in abi_pins(component.requirements.abi).items():
        if info.bus_id is not None or info.direction is not Direction.IN or name in streams:
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
    "Assembled",
    "COMPONENT",
    "Component",
    "Port",
    "STREAM_SPEC",
    "Stream",
    "StreamFifo",
    "StreamLink",
    "StreamSpec",
    "TopInput",
    "TopOutput",
    "assemble_streams",
]
