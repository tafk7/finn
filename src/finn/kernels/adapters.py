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

Each chain is a kernel with children (a ``Kernel`` binding no module): it
places its modules as kernel children (``InputGeneratorKernel``, ``VpcKernel``) named by
stage, each given its tensor and the two sides of its step, the beat sequences its ports
present, from which it derives its generics. Each child's streams and output word width
are a ``StageStreams`` (``stage_streams``, from the child's derived values), which the
channel checks like any other port's streams; and a ``Stage``: the child's module (read
from its netlist, ``finn.kernels.library.library_module``) and its ports' stream pins
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
from typing import ClassVar

from finn.core.space import (
    Param,
    Rejected,
    derived,
    reject,
    view,
)
from finn.dataflow.steps import Hop, Step, Steps, Unrealizable
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, Traversal
from finn.kernels.artifacts.module import Fragment, Leaf
from finn.kernels.base import Kernel, PortPins
from finn.kernels.input_generator import Generate, InputGeneratorKernel, generated
from finn.kernels.library import leaf_at_root
from finn.kernels.transport import StreamPins
from finn.kernels.vpc import VpcKernel
from finn.platform.capabilities import Platform


@dataclass(frozen=True)
class Stage:
    """A module inside a channel, with the stream pins of its two ports.

    ``module`` is its leaf, placed at ``label``: the node path of its kernel
    below whatever places it (``input_gen.input_gen`` below an adapter
    selector; ``adapter.input_gen.input_gen`` below the channel).
    """

    module: Leaf
    input: StreamPins
    output: StreamPins
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
    for hop in hops:
        for sequence in (hop.source, hop.sink):
            if not isinstance(sequence.form, Traversal):
                raise Unrealizable("no existing hardware adapter recipe carries Grouped forms")
            if any(not isinstance(rule, LevelEnd) for rule in sequence.markers):
                raise Unrealizable(
                    "no existing hardware adapter recipe carries held LevelDone markers"
                )
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


class StreamAdapter(Kernel):
    """A fixed chain of FinnLib modules carrying out the steps between a channel's two
    streams: the chain they take (``realize``), which its channel places for them.

    Each chain places its modules as kernel children named by stage
    (``input_gen``, ``vpc``, then ``input_gen_1``), each given the channel's tensor and
    the two sides of its step of the realization, which its ports present, on the
    channel's ``platform``. Its netlist, cycles, buffering and resources are its stages',
    by the base; it restates ``cycles`` as the base's, so that a channel counts its
    adapters' by their kind (``Channel.stage_cycles``). It presents no port: its channel
    checks its stages' streams (``stage_streams``) and wires its stages hop by hop
    (``stages``).
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
            return reject("adapter-steps", f"no FinnLib chain carries out the steps: {error}")

    def _named(self, name: str) -> RealizedStage | Rejected:
        realization: tuple[RealizedStage, ...] = self.realization
        kinds = [stage.kind for stage in realization]
        for index, stage in enumerate(realization):
            if _stage_name(kinds, index) == name:
                return stage
        return reject("adapter-steps", f"the steps' chain has no {name} stage")

    # The sides of every stage a chain can name; each chain reads its own.
    input_gen_arriving = _side("input_gen", "source")
    input_gen_leaving = _side("input_gen", "sink")
    input_gen_1_arriving = _side("input_gen_1", "source")
    input_gen_1_leaving = _side("input_gen_1", "sink")
    vpc_arriving = _side("vpc", "source")
    vpc_leaving = _side("vpc", "sink")

    @derived
    def stage_streams(self) -> tuple[StageStreams, ...]:
        """Each child's streams and output word width, its derived values, labelled as its
        ``Stage`` is. Each finite chain's native helper states its actual reads; this base
        places no stage."""
        return self._stage_streams()

    @view
    def stages(self) -> tuple[Stage, ...]:
        """Each child's module and its ports' stream pins, labelled below what holds the
        chain: the chain, then the stage (``vpc.vpc``). Each finite chain's native helper
        states its actual reads; this base places no stage."""
        return self._stages()

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        return ()

    def _stages(self) -> tuple[Stage, ...]:
        return ()

    @view
    def cycles(self) -> int | Rejected:
        """The most cycles any of its stages takes: the base's (the class docstring)."""
        return self._cycles()


class InputAdapter(StreamAdapter):
    """A chain after the transport (``ADAPTERS``): a reorder or markers first. Its input
    generators share one memory style, given (``ram_style``: its channel's one adapter
    style, ``Channel.adapter_ram_style``), which its channel ranks by the most bits any of
    them holds a frame (``frame_bits``)."""

    ram_style: str = Param()

    @derived
    def frame_bits(self) -> int:
        """The most bits any of its input generators holds a frame (each one's
        ``frame_bits``). Each finite chain reads its actual generators;
        this base places none."""
        return self._frame_bits()

    def _frame_bits(self) -> int:
        return 0


def _stage(
    label: str, given: PortPins, presented: PortPins, netlist: Fragment, owner: str
) -> Stage:
    """Pure stage assembly; callers read stream pins then the netlist in original order."""
    return Stage(leaf_at_root(netlist, owner), given.pins, presented.pins, label)


class vpc(StreamAdapter):
    modules = ("vpc",)
    vpc = VpcKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=StreamAdapter.vpc_arriving,
        leaving=StreamAdapter.vpc_leaving,
    )

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        return (
            StageStreams(
                "vpc.vpc", self.vpc.input.stream, self.vpc.output.stream, self.vpc.output.width
            ),
        )

    def _stages(self) -> tuple[Stage, ...]:
        return (
            _stage(
                "vpc.vpc",
                self.vpc.input.stream_pins,
                self.vpc.output.stream_pins,
                self.vpc.netlist,
                type(self.vpc).__qualname__,
            ),
        )


class input_gen(InputAdapter):
    modules = ("input_gen",)
    input_gen = InputGeneratorKernel(
        input_tensor=InputAdapter.tensor,
        arriving=InputAdapter.input_gen_arriving,
        leaving=InputAdapter.input_gen_leaving,
        platform=InputAdapter.platform,
        ram_style=InputAdapter.ram_style,
    )

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        return (
            StageStreams(
                "input_gen.input_gen",
                self.input_gen.input.stream,
                self.input_gen.output.stream,
                self.input_gen.output.width,
            ),
        )

    def _stages(self) -> tuple[Stage, ...]:
        return (
            _stage(
                "input_gen.input_gen",
                self.input_gen.input.stream_pins,
                self.input_gen.output.stream_pins,
                self.input_gen.netlist,
                type(self.input_gen).__qualname__,
            ),
        )

    def _frame_bits(self) -> int:
        return self.input_gen.frame_bits


class input_gen_vpc(InputAdapter):
    modules = ("input_gen", "vpc")
    input_gen = InputGeneratorKernel(
        input_tensor=InputAdapter.tensor,
        arriving=InputAdapter.input_gen_arriving,
        leaving=InputAdapter.input_gen_leaving,
        platform=InputAdapter.platform,
        ram_style=InputAdapter.ram_style,
    )
    vpc = VpcKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=StreamAdapter.vpc_arriving,
        leaving=StreamAdapter.vpc_leaving,
    )

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        return (
            StageStreams(
                "input_gen_vpc.input_gen",
                self.input_gen.input.stream,
                self.input_gen.output.stream,
                self.input_gen.output.width,
            ),
            StageStreams(
                "input_gen_vpc.vpc",
                self.vpc.input.stream,
                self.vpc.output.stream,
                self.vpc.output.width,
            ),
        )

    def _stages(self) -> tuple[Stage, ...]:
        return (
            _stage(
                "input_gen_vpc.input_gen",
                self.input_gen.input.stream_pins,
                self.input_gen.output.stream_pins,
                self.input_gen.netlist,
                type(self.input_gen).__qualname__,
            ),
            _stage(
                "input_gen_vpc.vpc",
                self.vpc.input.stream_pins,
                self.vpc.output.stream_pins,
                self.vpc.netlist,
                type(self.vpc).__qualname__,
            ),
        )

    def _frame_bits(self) -> int:
        return self.input_gen.frame_bits


class input_gen_vpc_input_gen(InputAdapter):
    modules = ("input_gen", "vpc", "input_gen")
    input_gen = InputGeneratorKernel(
        input_tensor=InputAdapter.tensor,
        arriving=InputAdapter.input_gen_arriving,
        leaving=InputAdapter.input_gen_leaving,
        platform=InputAdapter.platform,
        ram_style=InputAdapter.ram_style,
    )
    vpc = VpcKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=StreamAdapter.vpc_arriving,
        leaving=StreamAdapter.vpc_leaving,
    )
    input_gen_1 = InputGeneratorKernel(
        input_tensor=InputAdapter.tensor,
        arriving=InputAdapter.input_gen_1_arriving,
        leaving=InputAdapter.input_gen_1_leaving,
        platform=InputAdapter.platform,
        ram_style=InputAdapter.ram_style,
    )

    def _stage_streams(self) -> tuple[StageStreams, ...]:
        return (
            StageStreams(
                "input_gen_vpc_input_gen.input_gen",
                self.input_gen.input.stream,
                self.input_gen.output.stream,
                self.input_gen.output.width,
            ),
            StageStreams(
                "input_gen_vpc_input_gen.vpc",
                self.vpc.input.stream,
                self.vpc.output.stream,
                self.vpc.output.width,
            ),
            StageStreams(
                "input_gen_vpc_input_gen.input_gen_1",
                self.input_gen_1.input.stream,
                self.input_gen_1.output.stream,
                self.input_gen_1.output.width,
            ),
        )

    def _stages(self) -> tuple[Stage, ...]:
        return (
            _stage(
                "input_gen_vpc_input_gen.input_gen",
                self.input_gen.input.stream_pins,
                self.input_gen.output.stream_pins,
                self.input_gen.netlist,
                type(self.input_gen).__qualname__,
            ),
            _stage(
                "input_gen_vpc_input_gen.vpc",
                self.vpc.input.stream_pins,
                self.vpc.output.stream_pins,
                self.vpc.netlist,
                type(self.vpc).__qualname__,
            ),
            _stage(
                "input_gen_vpc_input_gen.input_gen_1",
                self.input_gen_1.input.stream_pins,
                self.input_gen_1.output.stream_pins,
                self.input_gen_1.netlist,
                type(self.input_gen_1).__qualname__,
            ),
        )

    def _frame_bits(self) -> int:
        return max(self.input_gen.frame_bits, self.input_gen_1.frame_bits)


INPUT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("input_gen",),
    ("input_gen", "vpc"),
    ("input_gen", "vpc", "input_gen"),
)
"""Every current input chain; stage and selection keys remain their module names."""

OUTPUT_ADAPTER: type[StreamAdapter] = vpc
ADAPTERS: dict[str, type[InputAdapter]] = {
    "input_gen": input_gen,
    "input_gen_vpc": input_gen_vpc,
    "input_gen_vpc_input_gen": input_gen_vpc_input_gen,
}


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
