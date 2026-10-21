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
  and sliding windows included, and marker synthesis; ``input_gen_vpc``, a reorder, then new lanes;
  ``input_gen_vpc_input_gen``, a reorder, new lanes, then markers (a ``vpc``
  carries none).

These are every shape a side of a plan can take (``OUTPUT_CHAINS``,
``INPUT_CHAINS``), so every realizable plan has exactly one candidate a side;
each is keyed by its modules, joined (``input_gen_vpc``).

``realize`` maps a plan onto modules. A reorder becomes an ``input_gen`` whose
frame, ``DIMS`` and ``COEFS`` are ``classify``'s; the marker step that follows
it merges into the same module, whose ``olst[d]`` bits close the levels
``DIMS[d:]``, split or grouped until every required level is one of them
(``finn.kernels.input_generator.generated``). Marker synthesis alone is an
``input_gen`` that passes its frames in order. A marker closing every beat is no
step: the connection ties it high. A width conversion is a ``vpc`` over vectors of
the two lane counts' least common multiple: its two sides carry the same elements in
whole beats, a common multiple of both lane counts, so the vectors are whole.

Each candidate is a kernel with children (a ``KernelSpace`` binding no module): it
places its modules as kernel children (``InputGeneratorKernel``, ``VpcKernel``) named by
stage, each given its tensor and the two sides of its step, the beat sequences its ports
present, from which it derives its generics. Each becomes a ``Stage``: the child's module
(read from its netlist, ``finn.kernels.library.library_module``) and its ports'
contracts, which the channel checks like any other port's, and which the channel wires
hop by hop, so an adapter presents no port of its own. Its netlist, cycles (the most
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
    composite,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.plan import Hop, Plan, Step, Unrealizable
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
    Decision; ``adapter.input_gen.input_gen`` below the channel).
    """

    module: Leaf
    input: StreamContract
    output: StreamContract
    label: str


# -- realizing a plan --------------------------------------------------------------------


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


def realize(plan: Plan) -> tuple[RealizedStage, ...]:
    """The FinnLib modules that carry out ``plan``, in order."""
    stages: list[RealizedStage] = []
    hops = list(plan.hops)
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


# -- the candidates of a channel's adapter Decision --------------------------------------


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
    """A fixed chain of FinnLib modules carrying out a channel's plan, or refusing it.

    Each candidate places its modules as kernel children named by stage
    (``input_gen``, ``vpc``, then ``input_gen_1``), each given the channel's tensor and
    the two sides of its step of the realization, which its ports present. An
    ``input_gen`` child owns its memory's ``ram_style``, on the channel's ``platform``.
    Its netlist, cycles, buffering and resources are its stages', by the base. It
    presents no port: its channel wires its stages hop by hop (``stages``).
    """

    id = "finn.channel.adapter"
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

    @view
    @declared_reads("realization")
    def stages(self) -> tuple[Stage, ...]:
        """Each child's module and its ports' contracts, labelled below the adapter
        Decision: its candidate (the chain), then the stage. Each chain declares what its
        own stages read (``_chain``); this base places no stage."""
        return self._stages()

    def _stages(self) -> tuple[Stage, ...]:
        """``stages``: each stage's child read by the name its chain gives it."""
        kinds = [stage.kind for stage in self.realization]
        chain = "_".join(type(self).modules)
        found: list[Stage] = []
        for index in range(len(kinds)):
            name = _stage_name(kinds, index)
            kernel = getattr(self, name)
            given: PortContract = kernel.input_contract
            presented: PortContract = kernel.output_contract
            found.append(
                Stage(library_module(kernel), given.contract, presented.contract, f"{chain}.{name}")
            )
        return tuple(found)

    exports = {
        NETLIST: KernelSpace.netlist,
        CYCLES: KernelSpace.cycles,
        BUFFERING: KernelSpace.buffering,
        RESOURCES: KernelSpace.resources,
    }


def _input_gen(name: str) -> InputGeneratorKernel:
    return InputGeneratorKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=getattr(StreamAdapter, f"{name}_arriving"),
        leaving=getattr(StreamAdapter, f"{name}_leaving"),
        platform=StreamAdapter.platform,
    )


def _vpc(name: str) -> VpcKernel:
    return VpcKernel(
        input_tensor=StreamAdapter.tensor,
        arriving=getattr(StreamAdapter, f"{name}_arriving"),
        leaving=getattr(StreamAdapter, f"{name}_leaving"),
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
    """The candidate placing ``modules`` in order, each child named by its stage. Its
    ``stages`` reads its children by the names its chain fixes, which it declares
    (``__reads__``: the realization, and each stage's module and ports' contracts)."""
    members: dict[str, object] = {"modules": modules}
    names = [_stage_name(modules, index) for index in range(len(modules))]
    for name, kind in zip(names, modules):
        members[name] = _input_gen(name) if kind == "input_gen" else _vpc(name)

    parts = ("netlist", "input_contract", "output_contract")

    @declared_reads("realization", *(f"{name}.{part}" for name in names for part in parts))
    def stages(self: StreamAdapter) -> tuple[Stage, ...]:
        return StreamAdapter._stages(self)

    members["stages"] = view(stages)
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
    "RealizedStage",
    "Stage",
    "StreamAdapter",
    "realize",
]
