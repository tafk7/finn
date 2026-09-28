# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream adapters: the hardware a stream places to carry out its plan.

A stream compares what its source presents with what its sink requires and
derives a plan (``finn.dataflow.plan``): reorders, width conversions and marker
synthesis. Its ``adapter`` Decision chooses the hardware that carries the plan
out. Each candidate is a fixed chain of FinnLib modules and refuses every plan
that chain does not realize, so at most one candidate survives a plan:

- ``input_gen``: a reorder, replay included, and marker synthesis
  (``InputGenAdapter``);
- ``vpc``: a width conversion (``WidthAdapter``);
- ``vpc_input_gen`` and ``input_gen_vpc``: a width conversion before or after a
  reorder or marker synthesis;
- ``input_gen_vpc_input_gen``: a reorder, new lanes, then markers (a ``vpc``
  carries none);
- ``vpc_input_gen_vpc`` and ``vpc_input_gen_vpc_input_gen``: a lane regroup
  through the common lane count, then markers.

These are every shape a plan can take, so every realizable plan has exactly one
candidate.

``realize`` maps a plan onto modules. A reorder becomes an ``input_gen`` whose
frame, ``DIMS`` and ``COEFS`` are ``classify``'s; the marker step that follows
it merges into the same module, whose ``olst[d]`` bits close the levels
``DIMS[d:]``, split or grouped until every required level is one of them.
Marker synthesis alone is an ``input_gen`` that passes its frames in order. A
width conversion is a ``vpc`` over vectors of the two lane counts' least common
multiple, which the stream must hold whole. Each module becomes a ``Stage``:
its build requirements and the contracts of its two ports, which the stream
checks like any other end.

FinnLib's ``inner_shuffle`` realizes one shape of lane regroup directly, but is
not a candidate yet: it emits undefined lanes under bursty input
(``finn.kernels.transpose``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm, prod
from typing import ClassVar

from finn.core.space import (
    Param,
    Rejected,
    Space,
    constraint,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.dataflow.plan import PLAN, Hop, Plan, Step, Unrealizable
from finn.dataflow.tensor import TENSOR, ScalarEncoding, Tensor
from finn.dataflow.traversal import LevelEnd, Presentation, Reorder
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Free, Reset, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.physical.contract import StreamContract
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker

INPUT_GEN_RAM_STYLES = ("auto", "distributed", "block", "ultra")


@dataclass(frozen=True)
class Stage:
    """A module inside a stream, with the contracts of its two ports; none when direct."""

    requirements: ModuleBuildRequirements | None = None
    input: StreamContract | None = None
    output: StreamContract | None = None
    name: str = ""


STAGE_SEMANTICS = default_semantics(Stage)
STAGES = default_semantics(tuple)


def clock_reset() -> tuple[Signal, Signal]:
    """FinnLib's native ``clk`` and synchronous active-high ``rst``."""
    return (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal(
            "rst",
            Direction.IN,
            1,
            Reset(active_low=False, synchronous=True, synchronous_to=("clk",)),
        ),
    )


def native(
    name: str, bits: int, endpoint: Endpoint, markers: tuple[StreamMarker, ...] = ()
) -> ReadyValidStream:
    """A FinnLib ``idat``/``ivld``/``irdy`` or ``odat``/``ovld``/``ordy`` port."""
    side = "i" if endpoint is Endpoint.TARGET else "o"
    return ReadyValidStream(
        name, bits, endpoint, f"{side}dat", f"{side}vld", f"{side}rdy", "clk", "rst", markers
    )


# -- the modules -------------------------------------------------------------------------


def input_gen_interfaces(word_bits: int, rank: int) -> tuple[ReadyValidStream, ReadyValidStream]:
    """Opaque words in; words out with the ``olst`` loop-completion vector."""
    return (
        native("input", word_bits, Endpoint.TARGET),
        native(
            "output",
            word_bits,
            Endpoint.INITIATOR,
            (StreamMarker("olst", MarkerKind.LOOP_END, rank),),
        ),
    )


def input_gen_requirements(
    *,
    word_bits: int,
    frame: int,
    dims: Sequence[int],
    coefs: Sequence[int],
    ram_style: str,
) -> ModuleBuildRequirements:
    """FinnLib ``input_gen``: per frame, the word at ``sum(index[i] * coefs[i])`` over ``dims``."""
    parameters = (
        ("COEFS", "'{" + ", ".join(map(str, coefs)) + "}"),
        ("D", len(dims)),
        ("DATA_WIDTH", word_bits),
        ("DIMS", "'{" + ", ".join(map(str, dims)) + "}"),
        ("FM_SIZE", frame),
        ("RAM_STYLE", f'"{ram_style}"'),
    )
    ports = input_gen_interfaces(word_bits, len(dims))
    abi = ModuleABIRequirements(
        FixedModuleName("input_gen"),
        (*clock_reset(), *(pin for port in ports for pin in port.pins())),
        tuple((key, str(value)) for key, value in parameters),
    )
    return ModuleBuildRequirements(
        "finnlib.input_generator",
        "1",
        parameters,
        abi,
        (CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",)),),
    )


def vpc_requirements(
    *, element_bits: int, lanes_in: int, lanes_out: int
) -> ModuleBuildRequirements:
    """FinnLib ``vpc``: vectors of ``lcm(lanes_in, lanes_out)`` elements, regrouped."""
    parameters = (
        ("N", lcm(lanes_in, lanes_out)),
        ("PAD_ZEROS", 1),
        ("PI", lanes_in),
        ("PO", lanes_out),
        ("RELAX_THROUGHPUT", 0),
        ("W", element_bits),
    )
    abi = ModuleABIRequirements(
        FixedModuleName("vpc"),
        (
            *clock_reset(),
            *native("input", lanes_in * element_bits, Endpoint.TARGET).pins(),
            *native("output", lanes_out * element_bits, Endpoint.INITIATOR).pins(),
        ),
        tuple((name, str(value)) for name, value in parameters),
    )
    return ModuleBuildRequirements(
        "finnlib.vpc",
        "1",
        parameters,
        abi,
        (CopiedSource("finnlib", "rtl/shape/vpc.sv", provides=("module:vpc",)),),
    )


# -- realizing a plan --------------------------------------------------------------------


@dataclass(frozen=True)
class Generate:
    """One ``input_gen``: per ``frame`` input beats, the nest ``dims`` stepping ``coefs``."""

    frame: int
    dims: tuple[int, ...]
    coefs: tuple[int, ...]

    @property
    def levels(self) -> tuple[LevelEnd, ...]:
        """``olst[d]`` closes the levels ``dims[d:]``."""
        return tuple(LevelEnd(prod(self.dims[depth:])) for depth in range(len(self.dims)))


@dataclass(frozen=True)
class Convert:
    """One ``vpc``: ``lanes_in`` elements a beat in, ``lanes_out`` out."""

    lanes_in: int
    lanes_out: int


@dataclass(frozen=True)
class RealizedStage:
    """A module and the presentations of its two ports."""

    module: Generate | Convert
    source: Presentation
    sink: Presentation

    @property
    def kind(self) -> str:
        return "input_gen" if isinstance(self.module, Generate) else "vpc"


def _closing(generate: Generate, required: Sequence[LevelEnd], input_beats: int) -> Generate:
    """``generate``'s nest split or grouped until it closes every ``required`` level."""
    frame, dims, coefs = generate.frame, list(generate.dims), list(generate.coefs)
    for level in sorted(required, key=lambda rule: rule.beats):
        if level.beats == 1 and dims[-1] != 1:
            # Every beat closes a level: an innermost loop of one.
            dims, coefs = [*dims, 1], [*coefs, 1]
            continue
        span = prod(dims)
        if level.beats > span:
            # Group whole frames under an outer level of frames.
            count = level.beats // span
            if level.beats % span or input_beats % (frame * count):
                raise Unrealizable(f"no input_gen frame closes {level.beats} beats")
            dims, coefs, frame = [count, *dims], [frame, *coefs], frame * count
            continue
        inner = 1
        for depth in range(len(dims) - 1, -1, -1):
            if inner == level.beats:
                break
            if inner * dims[depth] > level.beats:
                split = level.beats // inner
                if level.beats % inner or dims[depth] % split:
                    raise Unrealizable(f"no input_gen level closes every {level.beats} beats")
                dims[depth : depth + 1] = [dims[depth] // split, split]
                coefs[depth : depth + 1] = [coefs[depth] * split, coefs[depth]]
                break
            inner *= dims[depth]
    return Generate(frame, tuple(dims), tuple(coefs))


def _generated(hop: Hop, marked: Hop | None) -> RealizedStage:
    """An ``input_gen`` for a reorder, a marker step, or a reorder and its markers."""
    source = hop.source
    if hop.step is Step.REORDER:
        reorder = hop.reorder
        assert isinstance(reorder, Reorder)
        generate = Generate(reorder.frame_beats, reorder.dims, reorder.coefs)
    else:
        # Identity frames as long as the widest level it must close.
        widest = max(rule.beats for rule in hop.sink.markers)
        generate = Generate(widest, (widest,), (1,))
    required = (marked or hop).sink.markers if marked or hop.step is Step.MARKERS else ()
    generate = _closing(generate, required, source.form.beats)
    form = hop.sink.form
    offered = tuple(level for level in generate.levels if level.aligned(form))
    return RealizedStage(generate, source, Presentation(form, markers=offered))


def realize(plan: Plan) -> tuple[RealizedStage, ...]:
    """The FinnLib modules that carry out ``plan``, in order."""
    stages: list[RealizedStage] = []
    hops = list(plan.hops)
    index = 0
    while index < len(hops):
        hop = hops[index]
        index += 1
        if hop.step is Step.WIDTH:
            source, sink = hop.source, hop.sink
            elements = source.form.beats * source.form.lanes
            vector = lcm(source.form.lanes, sink.form.lanes)
            if elements % vector:
                raise Unrealizable(f"{elements} elements make no whole {vector}-element vectors")
            stages.append(RealizedStage(Convert(source.form.lanes, sink.form.lanes), source, sink))
            continue
        marked = None
        if hop.step is Step.REORDER and index < len(hops) and hops[index].step is Step.MARKERS:
            marked = hops[index]
            index += 1
        stages.append(_generated(hop, marked))
    return tuple(stages)


def _stage_name(kinds: Sequence[str], index: int) -> str:
    kind = kinds[index]
    earlier = kinds[:index].count(kind)
    return kind if not earlier else f"{kind}_{earlier}"


def _stage(stage: RealizedStage, element: ScalarEncoding, name: str, ram_style: str) -> Stage:
    source, sink, module = stage.source, stage.sink, stage.module
    word_in = source.form.lanes * element.bits
    word_out = sink.form.lanes * element.bits
    if isinstance(module, Convert):
        requirements = vpc_requirements(
            element_bits=element.bits, lanes_in=module.lanes_in, lanes_out=module.lanes_out
        )
        return Stage(
            requirements,
            StreamContract(native("input", word_in, Endpoint.TARGET), element, source.form),
            StreamContract(native("output", word_out, Endpoint.INITIATOR), element, sink.form),
            name,
        )
    requirements = input_gen_requirements(
        word_bits=word_in,
        frame=module.frame,
        dims=module.dims,
        coefs=module.coefs,
        ram_style=ram_style,
    )
    ports = input_gen_interfaces(word_in, len(module.dims))
    offered = {
        f"olst[{depth}]": level
        for depth, level in enumerate(module.levels)
        if level in sink.markers
    }
    return Stage(
        requirements,
        StreamContract(ports[0], element, source.form),
        StreamContract(ports[1], element, sink.form, markers=offered),
        name,
    )


# -- the candidates of a stream's adapter Decision ---------------------------------------


REALIZATION = default_semantics(tuple)


class StreamAdapter(Space):
    """A fixed chain of FinnLib modules carrying out a stream's plan, or refusing it.

    ``ram_style`` is the stream's choice for the memory of an ``input_gen``
    stage; it is read only by a chain that has one.
    """

    modules: ClassVar[tuple[str, ...]] = ()

    tensor: Tensor = Param(semantics=TENSOR)
    plan: Plan = Param(semantics=PLAN)
    ram_style: str = Param(required=False)

    @derived(semantics=REALIZATION)
    def realization(self) -> tuple[RealizedStage, ...] | Rejected:
        try:
            return realize(self.plan)
        except Unrealizable as error:
            return reject("adapter-plan", f"no FinnLib chain carries out the plan: {error}")

    @constraint
    def realizes(self) -> bool | Rejected:
        kinds = tuple(stage.kind for stage in self.realization)
        if kinds != self.modules:
            return reject(
                "adapter-plan",
                f"{' -> '.join(self.modules)} does not carry out {self.plan.describe()}, "
                f"which takes {' -> '.join(kinds)}",
            )
        return True

    @view(semantics=default_semantics(str), requires=(realizes,))
    def admitted(self) -> str:
        """Accepted exactly when this chain carries out the plan, whatever the memory."""
        return self.plan.describe()

    @view(semantics=STAGES, requires=(realizes,))
    def stages(self) -> tuple[Stage, ...]:
        element = self.tensor.element
        realization = self.realization
        kinds = [stage.kind for stage in realization]
        ram_style = self.ram_style if "input_gen" in kinds else ""
        return tuple(
            _stage(stage, element, _stage_name(kinds, index), ram_style)
            for index, stage in enumerate(realization)
        )


class InputGenAdapter(StreamAdapter):
    """One ``input_gen``: a reorder (replay included) and the markers it closes."""

    modules = ("input_gen",)


class WidthAdapter(StreamAdapter):
    """One ``vpc``: the same element order, another number of lanes a beat."""

    modules = ("vpc",)


class WidthReorderAdapter(StreamAdapter):
    """A ``vpc``, then an ``input_gen``: new lanes, then a reorder or markers."""

    modules = ("vpc", "input_gen")


class ReorderWidthAdapter(StreamAdapter):
    """An ``input_gen``, then a ``vpc``: a reorder at the source's lanes, then new lanes."""

    modules = ("input_gen", "vpc")


class RegroupAdapter(StreamAdapter):
    """``vpc``, ``input_gen``, ``vpc``: a lane regroup through the common lane count."""

    modules = ("vpc", "input_gen", "vpc")


class ReorderWidthMarkersAdapter(StreamAdapter):
    """``input_gen``, ``vpc``, ``input_gen``: markers after a reorder and new lanes.

    A ``vpc`` carries no markers, so the frame is closed after it.
    """

    modules = ("input_gen", "vpc", "input_gen")


class RegroupMarkersAdapter(StreamAdapter):
    """``vpc``, ``input_gen``, ``vpc``, ``input_gen``: a lane regroup, then markers."""

    modules = ("vpc", "input_gen", "vpc", "input_gen")


def buffers(plan: Plan) -> bool:
    """Whether the chain that carries out ``plan`` has an ``input_gen``."""
    try:
        return any(stage.kind == "input_gen" for stage in realize(plan))
    except Unrealizable:
        return False


__all__ = [
    "Convert",
    "Generate",
    "INPUT_GEN_RAM_STYLES",
    "InputGenAdapter",
    "RealizedStage",
    "RegroupAdapter",
    "RegroupMarkersAdapter",
    "ReorderWidthAdapter",
    "ReorderWidthMarkersAdapter",
    "STAGES",
    "STAGE_SEMANTICS",
    "Stage",
    "StreamAdapter",
    "WidthAdapter",
    "WidthReorderAdapter",
    "buffers",
    "clock_reset",
    "input_gen_interfaces",
    "input_gen_requirements",
    "native",
    "realize",
    "vpc_requirements",
]
