# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channel adapters: the hardware a channel places to carry out its plan.

A channel compares what its source presents with what its sink requires and
derives a plan (``finn.dataflow.plan``): reorders, width conversions and marker
synthesis, split at the channel's transport. Each side of the transport has its
adapter Decision, which chooses the hardware that carries out that side of the
plan. Each candidate is a fixed chain of FinnLib modules and refuses every plan
that chain does not realize, so at most one candidate survives a side:

- the output side, before the transport (``OUTPUT_ADAPTERS``): ``vpc``, a width
  conversion;
- the input side, after it (``ADAPTERS``): ``input_gen``, a reorder, replay
  included, and marker synthesis; ``input_gen_vpc``, a reorder, then new lanes;
  ``input_gen_vpc_input_gen``, a reorder, new lanes, then markers (a ``vpc``
  carries none).

These are every shape a side of a plan can take (``OUTPUT_CHAINS``,
``INPUT_CHAINS``), so every realizable plan has exactly one candidate a side;
each is keyed by its modules, joined (``input_gen_vpc``).

``realize`` maps a plan onto modules. A reorder becomes an ``input_gen`` whose
frame, ``DIMS`` and ``COEFS`` are ``classify``'s; the marker step that follows
it merges into the same module, whose ``olst[d]`` bits close the levels
``DIMS[d:]``, split or grouped until every required level is one of them.
Marker synthesis alone is an ``input_gen`` that passes its frames in order. A
marker closing every beat is no step: the connection ties it high. A
width conversion is a ``vpc`` over vectors of the two lane counts' least common
multiple, which the channel must hold whole. Each candidate places its modules
as kernel children (``InputGeneratorKernel``, ``VpcKernel``) named by stage,
and each becomes a ``Stage``: the child's module and the contracts of its two
ports, which the channel checks like any other end. Its cost reads the
realization alone, never a child's choices: ``beats``, the most beats any of its
ports carries, and ``held_bits``, the buffers its ``input_gen`` stages allocate
(``BUF_SIZE`` words each, read from the RTL: ``nest_geometry``). What it uses of
the device, ``resources``, is its children's own statements summed, which read their
memory styles.

FinnLib's ``replay_buffer`` is not wrapped: ``input_gen`` realizes every replay
it could. FinnLib's ``inner_shuffle`` realizes one shape of lane regroup
directly, but it is not a candidate: a channel realizes every lane regroup
through the common lane count (``vpc`` before the transport, ``input_gen_vpc``
after it), and ``TransposeKernel`` (``finn.kernels.transpose``) is
``inner_shuffle`` for a parent to place explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm, prod
from typing import Annotated, ClassVar

from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    Space,
    composite,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.plan import Hop, Plan, Step, Unrealizable
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, Reorder
from finn.kernels.artifacts.module import Leaf
from finn.kernels.input_generator import GeometryError, InputGeneratorKernel, nest_geometry
from finn.kernels.target import Platform
from finn.kernels.transport import StreamContract
from finn.kernels.utilization import RESOURCES_SEMANTICS, Resources, total
from finn.kernels.values.semantics import INTEGER_VECTOR, IntegerVector
from finn.kernels.vpc import VpcKernel


@dataclass(frozen=True)
class Stage:
    """A module inside a channel, with the contracts of its two ports.

    ``module`` is its leaf, placed at ``label``: the node path of its kernel
    below whatever places it (``input_gen.input_gen`` below an adapter
    Decision; ``adapter.input_gen.input_gen`` below the channel).
    """

    module: Leaf
    input: StreamContract
    output: StreamContract
    label: str


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
    """A module and the beat sequences of its two ports."""

    module: Generate | Convert
    source: BeatSequence
    sink: BeatSequence

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
    return RealizedStage(generate, source, BeatSequence(form, markers=offered))


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


# -- the candidates of a channel's adapter Decision --------------------------------------


@dataclass(frozen=True)
class InputGenFacts:
    """An ``input_gen`` stage's facts: its word, frame and nest."""

    word_bits: int
    frame: int
    dims: Annotated[IntegerVector, INTEGER_VECTOR]
    coefs: Annotated[IntegerVector, INTEGER_VECTOR]


@dataclass(frozen=True)
class VpcFacts:
    """A ``vpc`` stage's facts: its element and the two lane counts."""

    element_bits: int
    lanes_in: int
    lanes_out: int


class StreamAdapter(Space):
    """A fixed chain of FinnLib modules carrying out a channel's plan, or refusing it.

    Each candidate places its modules as kernel children named by stage
    (``input_gen``, ``vpc``, then ``input_gen_1``, ``vpc_1``), their facts
    derived from the realization. An ``input_gen`` child owns its memory's
    ``ram_style``, on the channel's ``platform``.
    """

    modules: ClassVar[tuple[str, ...]] = ()

    tensor: Tensor = Param()
    plan: Plan = Param()
    platform: Platform = Param()

    @derived
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

    # The kernels' convention for a refusal (``finn.kernels.configure.admission``).
    admission = ConstraintGroup(realizes)

    def _named(self, name: str) -> RealizedStage | Rejected:
        realization: tuple[RealizedStage, ...] = self.realization
        kinds = [stage.kind for stage in realization]
        for index, stage in enumerate(realization):
            if _stage_name(kinds, index) == name:
                return stage
        return reject("adapter-plan", f"the plan's chain has no {name} stage")

    def _generator(self, name: str) -> InputGenFacts | Rejected:
        stage = self._named(name)
        if isinstance(stage, Rejected):
            return stage
        module = stage.module
        assert isinstance(module, Generate)
        bits = stage.source.form.lanes * self.tensor.element.bits
        return InputGenFacts(bits, module.frame, module.dims, module.coefs)

    def _converter(self, name: str) -> VpcFacts | Rejected:
        stage = self._named(name)
        if isinstance(stage, Rejected):
            return stage
        module = stage.module
        assert isinstance(module, Convert)
        return VpcFacts(self.tensor.element.bits, module.lanes_in, module.lanes_out)

    # Cost, from the realization alone: a stage's memory style changes neither.
    @view
    def beats(self) -> int:
        """The most beats any of its stages' ports carries a frame."""
        return max(
            beats
            for stage in self.realization
            for beats in (stage.source.form.beats, stage.sink.form.beats)
        )

    @view
    def held_bits(self) -> int | Rejected:
        """The bits its ``input_gen`` stages hold: each its buffer, ``BUF_SIZE`` input
        words as the RTL derives it (``nest_geometry``), not one frame."""
        bits = self.tensor.element.bits
        held = 0
        for stage in self.realization:
            module = stage.module
            if not isinstance(module, Generate):
                continue
            try:
                words = nest_geometry(module.frame, module.dims, module.coefs).buffer_words
            except GeometryError as error:
                return reject("adapter-buffer", str(error))
            held += words * stage.source.form.lanes * bits
        return held

    @view(semantics=RESOURCES_SEMANTICS)
    def resources(self) -> Resources:
        """What its stages use of the device, each child's own statement, summed."""
        kinds = [stage.kind for stage in self.realization]
        return total(
            getattr(self, _stage_name(kinds, index)).resources for index in range(len(kinds))
        )

    # The facts of every stage a chain can name; each chain reads its own.
    @derived
    def input_gen_facts(self) -> InputGenFacts | Rejected:
        return self._generator("input_gen")

    @derived
    def input_gen_1_facts(self) -> InputGenFacts | Rejected:
        return self._generator("input_gen_1")

    @derived
    def vpc_facts(self) -> VpcFacts | Rejected:
        return self._converter("vpc")

    @derived
    def vpc_1_facts(self) -> VpcFacts | Rejected:
        return self._converter("vpc_1")

    @view(requires=(realizes,))
    def stages(self) -> tuple[Stage, ...]:
        """Each child's module and the contracts of its two ports."""
        element = self.tensor.element
        realization = self.realization
        kinds = [stage.kind for stage in realization]
        found: list[Stage] = []
        for index, stage in enumerate(realization):
            name = _stage_name(kinds, index)
            kernel = getattr(self, name)
            offered: tuple[tuple[str, LevelEnd], ...] = ()
            if isinstance(stage.module, Generate):
                offered = tuple(
                    (f"olst[{depth}]", level)
                    for depth, level in enumerate(stage.module.levels)
                    if level in stage.sink.markers
                )
            found.append(
                Stage(
                    kernel.module,
                    StreamContract(kernel.input.transport, element, stage.source.form),
                    StreamContract(
                        kernel.output.transport, element, stage.sink.form, markers=offered
                    ),
                    # Below the adapter Decision: its candidate (the chain), then the stage.
                    f"{'_'.join(type(self).modules)}.{name}",
                )
            )
        return tuple(found)


def _input_gen(facts: InputGenFacts) -> InputGeneratorKernel:
    return InputGeneratorKernel(
        word_bits=facts.word_bits,
        frame_words=facts.frame,
        dims=facts.dims,
        strides=facts.coefs,
        platform=StreamAdapter.platform,
    )


def _vpc(facts: VpcFacts) -> VpcKernel:
    return VpcKernel(
        element_bits=facts.element_bits, lanes_in=facts.lanes_in, lanes_out=facts.lanes_out
    )


OUTPUT_CHAINS: tuple[tuple[str, ...], ...] = (("vpc",),)
"""Every chain the output side of a plan can take: its width conversion."""

INPUT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("input_gen",),
    ("input_gen", "vpc"),
    ("input_gen", "vpc", "input_gen"),
)
"""Every chain the input side of a plan can take: a reorder or markers first."""


def _chain(modules: tuple[str, ...]) -> type[StreamAdapter]:
    """The candidate placing ``modules`` in order, each child named by its stage."""
    members: dict[str, object] = {"modules": modules}
    for index, kind in enumerate(modules):
        name = _stage_name(modules, index)
        facts = getattr(StreamAdapter, f"{name}_facts")
        members[name] = _input_gen(facts) if kind == "input_gen" else _vpc(facts)
    return composite("_".join(modules), members, base=StreamAdapter)


OUTPUT_ADAPTERS: dict[str, type[StreamAdapter]] = {
    "_".join(chain): _chain(chain) for chain in OUTPUT_CHAINS
}
"""The candidates of a channel's ``output_adapter`` Decision, by key."""

ADAPTERS: dict[str, type[StreamAdapter]] = {
    "_".join(chain): _chain(chain) for chain in INPUT_CHAINS
}
"""The candidates of a channel's ``adapter`` Decision (the input side), by key."""


__all__ = [
    "ADAPTERS",
    "INPUT_CHAINS",
    "OUTPUT_ADAPTERS",
    "OUTPUT_CHAINS",
    "Convert",
    "Generate",
    "InputGenFacts",
    "RealizedStage",
    "Stage",
    "StreamAdapter",
    "VpcFacts",
    "realize",
]
