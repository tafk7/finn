# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channel adapters: the hardware a channel places to carry out the steps between its
two streams.

A channel compares what its source presents with what its sink requires and
derives the steps between them (``finn.dataflow.steps``): reorders, width conversions
and marker synthesis, split at the channel's transport. Each side of the transport has
its adapter, the fixed chain of FinnLib modules its steps take:

- the output side, before the transport (``OUTPUT_ADAPTER``): ``vpc``, a width
  conversion, the one chain its steps can take, which the channel places where they
  convert a width;
- the input side, after it, the cases of the channel's ``adapter`` selector
  (``ADAPTERS``, each an ``InputAdapter``): ``input_gen``, a reorder, replay and sliding
  windows included, and marker synthesis; ``input_gen_vpc``, a reorder, then new lanes;
  ``input_gen_vpc_input_gen``, a reorder, new lanes, then markers (a ``vpc`` carries
  none).

These are every shape a side's steps can take (``OUTPUT_ADAPTER.modules``,
``INPUT_CHAINS``), so all realizable steps have exactly one chain a side; each is keyed
by its modules, joined (``input_gen_vpc``), which is how the channel picks its input
adapter: the kinds of ``realize``'s stages, joined (``Channel.input_chain``). A chain is
placed for the steps it realizes, so it states no rule that it does.

``realize`` maps the steps onto modules. A reorder becomes an ``input_gen`` whose
frame, ``DIMS`` and ``COEFS`` are ``classify``'s; the marker step that follows
it merges into the same module, whose ``olst[d]`` bits close the levels
``DIMS[d:]``, split or grouped until every required level is one of them
(``finn.kernels.input_generator.generated``). Marker synthesis alone is an
``input_gen`` that passes its frames in order. A marker closing every beat is no
step: the connection ties it high. A width conversion is a ``vpc`` over vectors of
the two lane counts' least common multiple: its two sides carry the same elements in
whole beats, a common multiple of both lane counts, so the vectors are whole.

Each chain is a kernel with children (a ``KernelSpace`` binding no module): it
places its modules as kernel children (``InputGeneratorKernel``, ``VpcKernel``) named by
stage, each given its tensor and the two sides of its step, the beat sequences its ports
present, from which it derives its generics. Each child's streams and output word width
are a ``StageStreams`` (``stage_streams``, from the child's derived values), which the
channel checks like any other port's streams; and a ``Stage``: the child's module (read
from its netlist, ``finn.kernels.library.library_module``) and its ports' contracts
(``stages``, a view), which the channel wires hop by hop, so an adapter presents no port
of its own. Its netlist, cycles (the most
beats any of its stages' ports carries), buffering (the buffers its ``input_gen`` stages
hold, ``BUF_SIZE`` words each, read from the RTL: ``nest_geometry``) and resources are its
stages' own, by the base's views.

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
from typing import ClassVar, TypeVar

from finn.core.space import (
    Param,
    Rejected,
    composite,
    derived,
    reject,
    view,
)
from finn.dataflow.steps import Hop, Step, Steps, Unrealizable
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.module import Leaf
from finn.kernels.base import (
    BUFFERING,
    CYCLES,
    NETLIST,
    RESOURCES,
    KernelSpace,
    PortContract,
)
from finn.kernels.declaration import declared_reads
from finn.kernels.input_generator import Generate, InputGeneratorKernel, generated
from finn.kernels.library import library_module
from finn.kernels.transport import StreamContract
from finn.kernels.vpc import VpcKernel
from finn.platform.capabilities import Platform


@dataclass(frozen=True)
class Stage:
    """A module inside a channel, with the contracts of its two ports.

    ``module`` is its leaf, placed at ``label``: the node path of its kernel
    below whatever places it (``input_gen.input_gen`` below an adapter
    selector; ``adapter.input_gen.input_gen`` below the channel).
    """

    module: Leaf
    input: StreamContract
    output: StreamContract
    label: str


@dataclass(frozen=True)
class StageStreams:
    """A stage inside a channel by its streams: the ``Stream`` it takes (``input``) and the
    one it presents (``output``), each its kernel's derived value, and its output's word
    width in bits (``width``), at ``label``, as its ``Stage`` is."""

    label: str
    input: Stream
    output: Stream
    width: int


# -- realizing the steps -----------------------------------------------------------------


@dataclass(frozen=True)
class Convert:
    """One ``vpc``: ``lanes_in`` elements a beat in, ``lanes_out`` out."""

    lanes_in: int
    lanes_out: int


@dataclass(frozen=True)
class RealizedStage:
    """A module and the two sides of its step: what arrives (``source``) and what it must
    present (``sink``: an ``input_gen``'s order and the markers it must close)."""

    module: Generate | Convert
    source: BeatSequence
    sink: BeatSequence

    @property
    def kind(self) -> str:
        return "input_gen" if isinstance(self.module, Generate) else "vpc"


def _generated(hop: Hop, marked: Hop | None) -> RealizedStage:
    """An ``input_gen`` for a reorder, a marker step, or a reorder and its markers: what
    arrives in the source's order, without its markers, and the order and markers of the
    step's end."""
    arriving = BeatSequence(hop.source.form)
    leaving = (marked or hop).sink
    return RealizedStage(generated(arriving, leaving), arriving, leaving)


def realize(steps: Steps) -> tuple[RealizedStage, ...]:
    """The FinnLib modules that carry out ``steps``, in order."""
    stages: list[RealizedStage] = []
    hops = list(steps.hops)
    index = 0
    while index < len(hops):
        hop = hops[index]
        index += 1
        if hop.step is Step.WIDTH:
            # Its sides carry no marker: a width conversion drops what arrives with one.
            source, sink = BeatSequence(hop.source.form), BeatSequence(hop.sink.form)
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


# -- a channel's adapters ----------------------------------------------------------------


def _side(name: str, side: str) -> BeatSequence:
    """A derived member of an adapter: one side of its stage ``name``'s step, ``source``
    (what arrives) or ``sink`` (what it presents), the beat sequence that stage's port is
    given. Name it in the class body (``vpc_arriving = _side("vpc", "source")``)."""

    def read(self: StreamAdapter) -> BeatSequence | Rejected:
        stage = self._named(name)
        if isinstance(stage, Rejected):
            return stage
        sequence: BeatSequence = getattr(stage, side)
        return sequence

    return derived(read)


class StreamAdapter(KernelSpace):
    """A fixed chain of FinnLib modules carrying out the steps between a channel's two
    streams: the chain they take (``realize``), which its channel places for them.

    Each chain places its modules as kernel children named by stage
    (``input_gen``, ``vpc``, then ``input_gen_1``), each given the channel's tensor and
    the two sides of its step of the realization, which its ports present, on the
    channel's ``platform``. Its netlist, cycles, buffering and resources are its stages',
    by the base. It presents no port: its channel checks its stages' streams
    (``stage_streams``) and wires its stages hop by hop (``stages``).
    """

    id = "finn.channel.adapter"
    modules: ClassVar[tuple[str, ...]] = ()

    tensor: Tensor = Param()
    steps: Steps = Param()
    platform: Platform = Param()

    @derived
    def realization(self) -> tuple[RealizedStage, ...] | Rejected:
        try:
            return realize(self.steps)
        except Unrealizable as error:
            return reject("adapter-plan", f"no FinnLib chain carries out the plan: {error}")

    def _named(self, name: str) -> RealizedStage | Rejected:
        realization: tuple[RealizedStage, ...] = self.realization
        kinds = [stage.kind for stage in realization]
        for index, stage in enumerate(realization):
            if _stage_name(kinds, index) == name:
                return stage
        return reject("adapter-plan", f"the plan's chain has no {name} stage")

    # The sides of every stage a chain can name; each chain reads its own.
    input_gen_arriving = _side("input_gen", "source")
    input_gen_leaving = _side("input_gen", "sink")
    input_gen_1_arriving = _side("input_gen_1", "source")
    input_gen_1_leaving = _side("input_gen_1", "sink")
    vpc_arriving = _side("vpc", "source")
    vpc_leaving = _side("vpc", "sink")

    @derived
    @declared_reads()
    def stage_streams(self) -> tuple[StageStreams, ...]:
        """Each child's streams and output word width, its derived values, labelled as its
        ``Stage`` is. Each chain declares what its own stages read (``_chain``); this base
        places no stage."""
        return self._stage_streams()

    @view
    @declared_reads("realization")
    def stages(self) -> tuple[Stage, ...]:
        """Each child's module and its ports' contracts, labelled below what holds the
        chain: the chain, then the stage (``vpc.vpc``). Each chain declares what its own
        stages read (``_chain``); this base places no stage."""
        return self._stages()

    def _named_stages(self) -> tuple[tuple[str, str], ...]:
        """Each stage's child's name, and its label below what holds the chain."""
        modules = type(self).modules
        chain = "_".join(modules)
        names = [_stage_name(modules, index) for index in range(len(modules))]
        return tuple((name, f"{chain}.{name}") for name in names)

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        """``stage_streams``: each stage's child read by the name its chain gives it."""
        found: list[StageStreams] = []
        for name, label in self._named_stages():
            kernel = getattr(self, name)
            found.append(
                StageStreams(
                    label, kernel.input_stream, kernel.output_stream, kernel.output_data_width
                )
            )
        return tuple(found)

    def _stages(self) -> tuple[Stage, ...]:
        """``stages``: each stage's child read by the name its chain gives it."""
        found: list[Stage] = []
        for name, label in self._named_stages():
            kernel = getattr(self, name)
            given: PortContract = kernel.input_contract
            presented: PortContract = kernel.output_contract
            found.append(Stage(library_module(kernel), given.contract, presented.contract, label))
        return tuple(found)

    exports = {
        NETLIST: KernelSpace.netlist,
        CYCLES: KernelSpace.cycles,
        BUFFERING: KernelSpace.buffering,
        RESOURCES: KernelSpace.resources,
    }


class InputAdapter(StreamAdapter):
    """A chain after the transport (``ADAPTERS``): a reorder or markers first. Its input
    generators share one memory style, given (``ram_style``: its channel's one adapter
    style, ``Channel.adapter_ram_style``), which its channel ranks by the most bits any of
    them holds a frame (``frame_bits``)."""

    ram_style: str = Param()

    @derived
    @declared_reads()
    def frame_bits(self) -> int:
        """The most bits any of its input generators holds a frame (each one's
        ``frame_bits``). Each chain declares what its own generators read (``_chain``);
        this base places none."""
        return self._frame_bits()

    def _frame_bits(self) -> int:
        """``frame_bits``: each input generator read by the name its chain gives it."""
        modules = type(self).modules
        names = [_stage_name(modules, index) for index in range(len(modules))]
        held = [
            getattr(self, name).frame_bits
            for name, kind in zip(names, modules)
            if kind == "input_gen"
        ]
        return max(held, default=0)


def _input_gen(name: str) -> InputGeneratorKernel:
    return InputGeneratorKernel(
        input_tensor=InputAdapter.tensor,
        arriving=getattr(InputAdapter, f"{name}_arriving"),
        leaving=getattr(InputAdapter, f"{name}_leaving"),
        platform=InputAdapter.platform,
        ram_style=InputAdapter.ram_style,
    )


def _vpc(name: str) -> VpcKernel:
    return VpcKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=getattr(StreamAdapter, f"{name}_arriving"),
        leaving=getattr(StreamAdapter, f"{name}_leaving"),
    )


INPUT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("input_gen",),
    ("input_gen", "vpc"),
    ("input_gen", "vpc", "input_gen"),
)
"""Every chain the input side's steps can take: a reorder or markers first."""


_Chain = TypeVar("_Chain", bound=StreamAdapter)


def _chain(modules: tuple[str, ...], base: type[_Chain]) -> type[_Chain]:
    """The chain placing ``modules`` in order, each child named by its stage, on ``base``.
    Its ``stage_streams`` and ``stages`` (and an input chain's ``frame_bits``) read its
    children by the names its chain fixes, which each declares (``__reads__``: each
    stage's streams and output word width, or its module and ports' contracts, or each
    input generator's frame bits)."""
    members: dict[str, object] = {"modules": modules}
    names = [_stage_name(modules, index) for index in range(len(modules))]
    for name, kind in zip(names, modules):
        members[name] = _input_gen(name) if kind == "input_gen" else _vpc(name)

    values = ("input_stream", "output_stream", "output_data_width")
    parts = ("netlist", "input_contract", "output_contract")

    @declared_reads(*(f"{name}.{part}" for name in names for part in values))
    def stage_streams(self: StreamAdapter) -> tuple[StageStreams, ...]:
        return StreamAdapter._stage_streams(self)

    @declared_reads("realization", *(f"{name}.{part}" for name in names for part in parts))
    def stages(self: StreamAdapter) -> tuple[Stage, ...]:
        return StreamAdapter._stages(self)

    members["stage_streams"] = derived(stage_streams)
    members["stages"] = view(stages)
    if issubclass(base, InputAdapter):
        generators = [name for name, kind in zip(names, modules) if kind == "input_gen"]

        @declared_reads(*(f"{name}.frame_bits" for name in generators))
        def frame_bits(self: InputAdapter) -> int:
            return InputAdapter._frame_bits(self)

        members["frame_bits"] = derived(frame_bits)
    return composite("_".join(modules), members, base=base)


OUTPUT_ADAPTER: type[StreamAdapter] = _chain(("vpc",), StreamAdapter)
"""A channel's output adapter: the one chain the output side's steps can take, its width
conversion, placed where they convert a width."""

ADAPTERS: dict[str, type[InputAdapter]] = {
    "_".join(chain): _chain(chain, InputAdapter) for chain in INPUT_CHAINS
}
"""The cases of a channel's ``adapter`` selector (the input side), by key: the kinds of the
stages each places, joined, as the channel's ``input_chain`` names them."""


__all__ = [
    "ADAPTERS",
    "INPUT_CHAINS",
    "OUTPUT_ADAPTER",
    "Convert",
    "InputAdapter",
    "RealizedStage",
    "Stage",
    "StageStreams",
    "StreamAdapter",
    "realize",
]
