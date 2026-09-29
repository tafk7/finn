# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Composite kernels: kernels of kernels, wired through their streams.

A composite places kernels, Decisions over kernels and the streams between
them; ``structure`` wires its children's modules through its streams
(``netlist``) into one generated module whose ports are its boundary streams.
A ``Design`` is a composite at the top.

The module has one clock: ``ap_clk`` and the active-low reset ``ap_rst_n``,
plus ``ap_clk2x`` when a child needs an aligned double-rate clock. ``netlist``
drives each child clock and reset pin by its declared role, never by its name;
holds what each kernel ties off (``Members(TIEOFFS)``); and exports control
buses (``Members(EXPORTED)``) at the module boundary. Wiring the module's own
instance into a design is its consumer's.

A composite may itself be placed in another. It then sits on its parent's
streams through reference inputs, each paired in ``boundaries`` with the
internal stream that is its boundary there (``{"x_stream": "activations"}``),
and exports, under ``PORT``, that stream's boundary contract for each: the
parent's stream plans and adapts to the composite as to any kernel. Its
``fused`` Decision says what it becomes in its parent:

- fused, one module (``MODULE``), which the parent nests as a child; a fused
  composite exposing a control bus is refused (``composite-control``), as
  its parent does not export a child's bus yet;
- otherwise its parts (``PARTS``): the parent places its modules, streams,
  tie-offs and control buses in its own module, each boundary stream spliced
  with the parent's stream it sits on, each bus exported as
  ``<composite>_<port>``.

The two are the same wiring with a different module hierarchy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

from finn.core.space import (
    Decision,
    Located,
    Members,
    Rejected,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.kernels.adapters import Stage
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
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
from finn.kernels.base import MODULE, MODULE_REQUIREMENTS, TIEOFFS, Kernel, Tieoffs
from finn.kernels.control import EXPORTED, Exported, top_bus
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import StreamContract
from finn.kernels.physical.lowering import lower_module_structure, nested
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.physical.validation import abi_pins
from finn.kernels.streams import CLOCK, CLOCK2X, CONNECTION, RESET, Connection


@dataclass(frozen=True)
class Composed:
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements


COMPOSED = default_semantics(Composed)


@dataclass(frozen=True)
class Parts:
    """A composite's netlist inputs, for its parent to place in its own module.

    ``boundaries`` pairs each reference input through which the composite sits
    on a stream of its parent (``x_stream``) with the internal stream that is
    its boundary there (``activations``). Names are the composite's own.
    """

    modules: tuple[Located[ModuleBuildRequirements], ...] = ()
    streams: tuple[Located[Connection], ...] = ()
    tieoffs: tuple[Located[Tieoffs], ...] = ()
    controls: tuple[Located[tuple[Exported, ...]], ...] = ()
    boundaries: tuple[tuple[str, str], ...] = ()


PARTS_SEMANTICS = default_semantics(Parts)
PARTS = ViewKey("parts", PARTS_SEMANTICS)
"""A composite's parts, exported instead of its module when its parent places them."""


def _instance(node: str | None) -> str | None:
    """``u_<node>``; a candidate of a Decision (``memory.memstream``) joins with ``_``."""
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
    parts: Sequence[Located[Parts]] = (),
    *,
    module: str,
    producer: ProducerIdentity,
) -> Composed | Rejected:
    """Wire the composite's modules through its streams and control buses.

    Each module is instantiated as ``u_<node>``, each stage of a stream as
    ``u_<stream>_<stage>`` (``u_activations_input_gen``, ``u_weights_fifo``). A
    stream end presented by a kernel's port belongs to that kernel's instance,
    and a module placed inside a stream is one of that stream's stages. A
    composed child module is nested (``finn.kernels.physical.lowering.nested``).
    Every clock and reset pin is driven by its role (a free clock from
    ``ap_clk``, a clock at twice another from ``ap_clk2x``, a reset from
    ``ap_rst_n``), every exported control bus is wired through to its top port,
    and every tied input is held by its kernel's ``Tieoffs``. An input nothing
    drives is refused, as is a defect no single stream can see, such as a
    clock-domain conflict.

    A child placed as ``parts`` joins this module: its modules, streams,
    tie-offs and control buses are named below it (``u_mm_compute_packed``,
    its buses' ports ``mm_s_axilite``), and each of its boundary streams is
    spliced with the stream of this composite it sits on there, the two
    stream's stages in order between the outer end and the inner one.
    """
    try:
        modules, streams, tieoffs, controls = merge_parts(
            list(modules), list(streams), list(tieoffs), list(controls), parts
        )
    except ValueError as error:
        return reject("stream-composition", str(error))
    # A module placed inside a stream (an adapter's or a FIFO's) is one of its
    # stages, wired by the stream as u_<stream>_<stage>.
    inside = tuple(str(item.node) + "." for item in streams)
    modules = [item for item in modules if not str(item.node).startswith(inside)]
    tieoffs = [item for item in tieoffs if not str(item.node).startswith(inside)]
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
            {str(_instance(item.node)): nested(item.value) for item in modules},
            connections,
            {str(_instance(item.node)): item.value for item in tieoffs},
            [exported for item in controls for exported in item.value],
            module,
            producer,
        )
    except ValueError as error:
        return reject("stream-composition", str(error))


def _below(child: str, node: str | None) -> str | None:
    return None if node is None else f"{child}.{node}" if node else child


def _tagged(stages: Sequence[Stage], stream: str) -> tuple[Stage, ...]:
    """Stages named by the stream that placed them, once they share another's connection."""
    return tuple(stage if stage.stream else replace(stage, stream=stream) for stage in stages)


def merge_parts(
    modules: list[Located[ModuleBuildRequirements]],
    streams: list[Located[Connection]],
    tieoffs: list[Located[Tieoffs]],
    controls: list[Located[tuple[Exported, ...]]],
    parts: Sequence[Located[Parts]],
) -> tuple[
    list[Located[ModuleBuildRequirements]],
    list[Located[Connection]],
    list[Located[Tieoffs]],
    list[Located[tuple[Exported, ...]]],
]:
    """Every child's parts named below it; each child boundary spliced with its stream here.

    Raises ``ValueError`` for a child boundary that sits on no stream here.
    """
    boundary: dict[tuple[str, str], tuple[str, Connection]] = {}
    for item in parts:
        child, part = str(item.node), item.value
        at = {stream: name for name, stream in part.boundaries}
        modules += [Located(_below(child, m.node), m.member, m.value) for m in part.modules]
        tieoffs += [Located(_below(child, t.node), t.member, t.value) for t in part.tieoffs]
        controls += [
            Located(
                _below(child, c.node),
                c.member,
                tuple(
                    replace(
                        e,
                        node=str(_below(child, e.node)),
                        port=f"{child.replace('.', '_')}_{e.port}",
                    )
                    for e in c.value
                ),
            )
            for c in part.controls
        ]
        for located in part.streams:
            name, value = str(_below(child, located.node)), located.value
            renamed = replace(
                value,
                source_owner=_below(child, value.source_owner),
                sink_owner=_below(child, value.sink_owner),
                stages=_tagged(value.stages, name),
            )
            reference = at.get(str(located.node))
            if reference is not None and None in (value.source_owner, value.sink_owner):
                boundary[(child, reference)] = (name, renamed)
            else:
                streams.append(Located(name, located.member, renamed))
    spliced: list[Located[Connection]] = []
    for located in streams:
        outer = located.value
        name = str(located.node)
        inner = boundary.pop((str(outer.sink_owner), outer.sink_input), None)
        if inner is not None:
            # The child's input boundary: the outer stream's stages, then the inner one's.
            outer = replace(
                outer,
                sink_owner=inner[1].sink_owner,
                sink=inner[1].sink,
                sink_input=inner[1].sink_input,
                stages=(*_tagged(outer.stages, name), *inner[1].stages),
            )
        inner = boundary.pop((str(outer.source_owner), outer.source_input), None)
        if inner is not None:
            # The child's output boundary: the inner stream's stages, then the outer one's.
            outer = replace(
                outer,
                source_owner=inner[1].source_owner,
                source=inner[1].source,
                source_input=inner[1].source_input,
                stages=(*inner[1].stages, *_tagged(outer.stages, name)),
            )
        spliced.append(Located(located.node, located.member, outer))
    if boundary:
        unplaced = ", ".join(f"{child}.{reference}" for child, reference in boundary)
        raise ValueError(f"the boundaries {unplaced} of a placed composite sit on no stream")
    return modules, spliced, tieoffs, controls


def _wire(
    placed: dict[str, ModuleBuildRequirements],
    connections: list[tuple[str, Connection]],
    tieoffs: dict[str, Tieoffs],
    exported: list[Exported],
    module: str,
    producer: ProducerIdentity,
) -> Composed:
    staged = [
        (_stage_instance(name, stage), stage)
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
            instance = _stage_instance(name, stage)
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


def _stage_instance(connection: str, stage: Stage) -> str:
    """``u_<stream>_<stage>``, by the stream that placed it."""
    return f"u_{(stage.stream or connection).replace('.', '_')}_{stage.name}"


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
        else:
            raise ValueError(f"{instance}.{name}: an input outside every stream has no driver")
    for name in tieoffs.unused:
        composition.dispose(instance, name, "tied off")


class Composite(Kernel):
    """Its children's modules wired through its streams: one module, or its parent's parts."""

    id = "finn.composite"
    version = "1"

    # Reference input -> the internal stream that is the composite's boundary on it.
    boundaries: ClassVar[Mapping[str, str]] = {}

    modules = Members(MODULE)
    streams = Members(CONNECTION)
    tied = Members(TIEOFFS)
    controls = Members(EXPORTED)
    flattened = Members(PARTS)

    def stem(self) -> str:
        """The generated module's name stem."""
        return "finn_" + type(self).__name__.lower()

    def producer_identity(self) -> ProducerIdentity:
        return ProducerIdentity("finn." + type(self).__name__.lower(), "1")

    @derived
    def placed(self) -> bool:
        """Whether it sits on a stream of a parent."""
        family = type(self)
        return any(self.present(getattr(family, reference)) for reference in family.boundaries)

    fused: bool = Decision(values=(True, False), when=placed)

    @derived
    def parted(self) -> bool:
        return not self.fused

    @constraint
    def seated(self) -> bool | Rejected:
        """Each parent stream it sits on carries its boundary stream's tensor."""
        for reference, stream in type(self).boundaries.items():
            if not self.present(getattr(type(self), reference)):
                continue
            outer, inner = getattr(self, reference).tensor, getattr(self, stream).tensor
            if outer != inner:
                return reject(
                    "composite-tensor",
                    f"{reference} carries {outer.shape} {outer.element.datatype_name}; "
                    f"the boundary {stream} carries {inner.shape} {inner.element.datatype_name}",
                )
        return True

    @view(
        semantics=COMPOSED,
        requires=(Kernel.admission, seated, modules, streams, tied, controls, flattened),
    )
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.tied,
            self.controls,
            self.flattened,
            module=self.stem(),
            producer=self.producer_identity(),
        )

    @view(semantics=MODULE_REQUIREMENTS, requires=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure.requirements

    @constraint
    def uncontrolled(self) -> bool | Rejected:
        """A fused composite exposes no control bus: its parent does not export one yet."""
        if any(item.value for item in self.controls):
            return reject(
                "composite-control",
                "a fused composite's control bus is not exported through its parent; "
                "place it unfused",
            )
        return True

    # What it exports to a parent: its module when fused, its parts otherwise.
    module_export = View(build_requirements, when=fused, requires=(uncontrolled,))

    @view(
        semantics=PARTS_SEMANTICS,
        requires=(Kernel.admission, seated, modules, streams, tied, controls, flattened),
        when=parted,
    )
    def parts(self) -> Parts | Rejected:
        try:
            modules, streams, tieoffs, controls = merge_parts(
                list(self.modules),
                list(self.streams),
                list(self.tied),
                list(self.controls),
                self.flattened,
            )
        except ValueError as error:
            return reject("stream-composition", str(error))
        return Parts(
            tuple(modules),
            tuple(streams),
            tuple(tieoffs),
            tuple(controls),
            tuple(type(self).boundaries.items()),
        )

    exports = {MODULE: module_export, PARTS: parts}


class Design(Composite):
    """A composite at the top: its boundary streams are its module's ports."""

    id = "finn.design"
    version = "1"


__all__ = [
    "COMPOSED",
    "Composed",
    "Composite",
    "Design",
    "PARTS",
    "PARTS_SEMANTICS",
    "Parts",
    "merge_parts",
    "netlist",
]
