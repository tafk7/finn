# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by its loop extents (``dims``). Zero strides repeat words,
and so do loops stepping the same words (a sliding window). This declaration
admits finite traversals wholly within each frame, which is all a window needs:
the module frees the words no later loop step reads, so a window holds a few
rows of its image, not the frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.

It presents the sequences it is given on its ports: ``arriving`` on its input, and on
its output ``leaving``'s order with every level its nest closes there (``olst``'s bits,
``Generate.levels``). It derives its generics from them (``generated``): ``DATA_WIDTH``
its input's word, and the nest (``FM_SIZE``, ``DIMS``, ``COEFS``) the reorder between
their orders (``classify``), or, for the same order, frames as long as the widest marker
``leaving`` requires, split or grouped until it closes each of ``leaving``'s markers. On
a channel,
``input_gen`` is a stage of the channel's adapter (``finn.kernels.adapters``), which
gives it the two sides of its step of the channel's plan. Its cycles are its ports'
beats, as every leaf's, and its buffering its buffer, ``BUF_SIZE`` input words.
Its buffer's ``ram_style`` is its choice: the explicit styles, ordered by the most the
buffer can hold (``frame_bits``, a frame), then ``auto``, which states no resources
(``finn.kernels.utilization.memory_styles``); ``ultra`` needs the ``platform``'s
UltraRAM, a constraint of the kernel (``uram-absent``). The buffer starts empty, so no initial
contents are asked of it.

What the buffer does is read from the RTL, not copied:
``nest_geometry`` elaborates FinnLib's ``input_gen.sv`` with slang at a nest's
parameters and reads the constants it derives (``BUF_SIZE``, ``MAX_OCCUPANCY``,
``R_FLAG``, ``TERMINAL_RP_INC``, ``TERMINAL_FP_INC``). Only the runtime rule
stays here: ``nest_buffer`` steps the read and free pointers by those increments
over a frame, as the RTL's nest counters select them, and states which word each
output beat reads and how many words are freed after it. The buffer accepts a
word while fewer than ``BUF_SIZE - 1`` accepted words are unfreed. FinnLib is the
``finnlib`` resource as FINN resolves it (``finn.resources``), the copy every
emitted module takes this source from. A cost query reads it and never fetches it:
where no local copy is (an override, a path, a cache), ``nest_geometry`` refuses
with the resource's error, which names it, rather than doing network I/O. The
constants are evaluated once per nest and source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from finn import resources
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.plan import Unrealizable
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import Adaptation, BeatSequence, LevelEnd, classify
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.rtl import Declined, evaluate
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace
from finn.kernels.library import LibraryModule
from finn.kernels.port import LoopEnds, data_width, native_stream
from finn.kernels.utilization import (
    RESOURCES_SEMANTICS,
    Fit,
    auto_unstated,
    memory,
    memory_styles,
    uram_constraint,
)
from finn.kernels.values.semantics import INTEGER_VECTOR, IntegerVector
from finn.platform.capabilities import Fabric, Platform, Resources

SOURCE = CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",))


def _vector(values: IntegerVector) -> str:
    return "'{" + ", ".join(map(str, values)) + "}"


def input_gen_resources(
    *, buf_size: int, data_width: int, d: int, ram_style: str, fabric: Fabric
) -> Resources:
    """FinnLib ``input_gen``: its buffer, BUF_SIZE words of DATA_WIDTH bits, simple dual
    port, in RAM_STYLE on ``fabric`` (``nest_geometry`` reads BUF_SIZE from the RTL), and
    its read and output registers, two a bit. The nest's D counters and pointers are a
    ``Fit`` over their address bits."""
    address = d * max(buf_size - 1, 1).bit_length()
    return memory(buf_size, data_width, ram_style, fabric=fabric) + Resources(
        lut=_INPUT_GEN_LUT.at(address), ff=2 * data_width + _INPUT_GEN_FF.at(address)
    )


# Feature: the loops' address bits, D * clog2(BUF_SIZE).
_INPUT_GEN_LUT = Fit(9.9, (3.07,))
_INPUT_GEN_FF = Fit(12.7, (2.31,))


# -- the nest, from the two sequences ---------------------------------------------------------


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


def generated(arriving: BeatSequence, leaving: BeatSequence) -> Generate:
    """The nest an ``input_gen`` walks to present ``leaving``'s order and markers from
    ``arriving``: the reorder between their orders (``classify``: its frame, extents and
    strides), or, for the same order, identity frames as long as the widest marker
    ``leaving`` requires; split or grouped until it closes each of ``leaving``'s markers.
    ``Unrealizable`` where no nest does."""
    source, sink = arriving.form, leaving.form
    if source != sink:
        verdict = classify(source, sink)
        reorder = verdict.reorder
        if verdict.adaptation is not Adaptation.REORDER or reorder is None:
            raise Unrealizable(f"no input_gen reorders them: {verdict.detail}")
        generate = Generate(reorder.frame_beats, reorder.dims, reorder.coefs)
    else:
        if not leaving.markers:
            raise Unrealizable("the same order and no marker: nothing for an input_gen to do")
        widest = max(rule.beats for rule in leaving.markers)
        generate = Generate(widest, (widest,), (1,))
    return _closing(generate, leaving.markers, source.beats)


def presented(generate: Generate, leaving: BeatSequence) -> BeatSequence:
    """What an ``input_gen`` walking ``generate`` presents of ``leaving``: its order, with
    every level its nest closes that the order aligns with."""
    form = leaving.form
    return BeatSequence(
        form, markers=tuple(level for level in generate.levels if level.aligned(form))
    )


class InputGeneratorKernel(KernelSpace):
    """FinnLib ``input_gen``: its tensor's words in another order, with its loop ends; see
    the module docstring."""

    id = "finnlib.input_generator"
    clock_domain = NATIVE_CLOCKING

    # The tensor whose words it reorders, a fact; what arrives of it, and the order and
    # markers it presents (``leaving``: the markers it must close).
    input_tensor: Tensor = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()
    platform: Platform = Param()

    @derived
    def output_tensor(self) -> Tensor:
        """What it moves its input to: the same tensor."""
        tensor: Tensor = self.input_tensor
        return tensor

    @derived
    def nest(self) -> Generate | Rejected:
        """The nest it walks, from the two sequences (``generated``)."""
        try:
            return generated(self.arriving, self.leaving)
        except Unrealizable as error:
            return reject("input-generator-nest", str(error))

    @derived
    def presenting(self) -> BeatSequence:
        """What its output presents: ``leaving``'s order, with every level its nest closes
        there (``presented``)."""
        return presented(self.nest, self.leaving)

    @derived
    def loop_ends(self) -> LoopEnds:
        """``olst``: one bit per loop, closing that loop and every inner one."""
        return LoopEnds("olst", self.nest.levels)

    input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)
    output = native_stream("output", Endpoint.INITIATOR, output_tensor, presenting, loop_ends)
    if TYPE_CHECKING:  # members its ports lower to, which it reads
        input_stream: ClassVar[Stream]

    @derived
    def word_bits(self) -> int | Rejected:
        """DATA_WIDTH: its input port's word."""
        return data_width(InputGeneratorKernel.input, self.input_stream)

    @derived
    def frame_words(self) -> int:
        """FM_SIZE: the input words of a frame."""
        return self.nest.frame

    @derived(semantics=INTEGER_VECTOR)
    def dims(self) -> IntegerVector:
        """DIMS: the nest's loop extents, outer first."""
        return self.nest.dims

    @derived(semantics=INTEGER_VECTOR)
    def strides(self) -> IntegerVector:
        """COEFS: the input words each loop steps."""
        return self.nest.coefs

    @constraint
    def traversal_supported(self) -> bool | Rejected:
        """The nest's arithmetic fits the RTL's native widths (a nest from two sequences
        is otherwise one it walks: positive extents and strides within its frame)."""
        values = (self.frame_words, *self.dims, *self.strides)
        if self.word_bits > 0xFFFFFFFF or any(value > 0x7FFFFFFF for value in values):
            return reject(
                "input-generator-range", "loop arithmetic must fit native signed 32-bit increments"
            )
        return True

    @derived
    def frame_bits(self) -> int:
        """The most the buffer can hold: a frame's words. Its order reads no ``BUF_SIZE``,
        which the RTL elaborates and which may be declined (its statement only)."""
        return self.frame_words * self.word_bits

    ram_style: str = Decision(domain=memory_styles(frame_bits))
    uram_present = uram_constraint(platform, ram_style, ("ultra",))

    @view
    def buffer_words(self) -> int | Rejected:
        """BUF_SIZE, as the RTL derives it (``nest_geometry``)."""
        try:
            return nest_geometry(self.frame_words, self.dims, self.strides).buffer_words
        except GeometryError as error:
            return reject("input-generator-geometry", str(error))

    @view
    def held_bits(self) -> int:
        """What it holds: its buffer, ``BUF_SIZE`` input words, not one frame."""
        words: int = self.buffer_words
        return words * self.word_bits

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        if self.ram_style == "auto":
            return auto_unstated("the input_gen buffer")
        return input_gen_resources(
            buf_size=self.buffer_words,
            data_width=self.word_bits,
            d=len(self.dims),
            ram_style=self.ram_style,
            fabric=self.platform.fabric,
        )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "COEFS": _vector(self.strides),
            "D": len(self.dims),
            "DATA_WIDTH": self.word_bits,
            "DIMS": _vector(self.dims),
            "FM_SIZE": self.frame_words,
            "RAM_STYLE": f'"{self.ram_style}"',
        }

    module = LibraryModule("input_gen", parameters=parameters, sources=(SOURCE,))


# -- the buffer, as the RTL derives it -------------------------------------------------------


class GeometryError(Refusal):
    """A nest whose buffer could not be read from the RTL: slang declined, named. One
    ``input-generator-geometry`` rejection, owned by ``input_gen``, the RTL module whose
    elaboration declined."""

    def __init__(self, message: str) -> None:
        super().__init__(
            (Finding(FindingKind.REJECTION, "input-generator-geometry", "input_gen", message),),
            message,
        )


@dataclass(frozen=True)
class NestGeometry:
    """``input_gen``'s elaboration-time constants for one nest, as its RTL evaluates them.

    ``buffer_words`` is ``BUF_SIZE``; per level ``0 … D`` (``D``: the default
    innermost advance), ``frees`` is ``R_FLAG``, ``read_steps`` is
    ``TERMINAL_RP_INC`` and ``free_steps`` the words ``TERMINAL_FP_INC`` frees (the
    RTL stores it negated).
    """

    buffer_words: int
    max_occupancy: int
    frees: tuple[bool, ...]
    read_steps: tuple[int, ...]
    free_steps: tuple[int, ...]

    @property
    def capacity(self) -> int:
        """The accepted words it holds unfreed at most: ``BUF_SIZE - 1``."""
        return self.buffer_words - 1


_CONSTANTS = ("BUF_SIZE", "MAX_OCCUPANCY", "R_FLAG", "TERMINAL_RP_INC", "TERMINAL_FP_INC")


@cache
def _evaluated(
    source: Path, frame_words: int, dims: tuple[int, ...], strides: tuple[int, ...]
) -> NestGeometry:
    # DATA_WIDTH and RAM_STYLE enter none of the nest computations: one word bit.
    binding = (
        ("COEFS", _vector(strides)),
        ("D", str(len(dims))),
        ("DATA_WIDTH", "1"),
        ("DIMS", _vector(dims)),
        ("FM_SIZE", str(frame_words)),
    )
    found = evaluate((source,), "input_gen", binding, _CONSTANTS)
    if isinstance(found, Declined):
        raise GeometryError(f"input_gen {dict(binding)}: {found}")
    buffer, occupancy, frees, reads, negated = (found[name] for name in _CONSTANTS)
    assert isinstance(buffer, int) and isinstance(occupancy, int)
    assert isinstance(frees, tuple) and isinstance(reads, tuple) and isinstance(negated, tuple)
    return NestGeometry(
        buffer, occupancy, tuple(map(bool, frees)), reads, tuple(-step for step in negated)
    )


def nest_geometry(frame_words: int, dims: Sequence[int], strides: Sequence[int]) -> NestGeometry:
    """The constants FinnLib's ``input_gen`` derives for ``FM_SIZE``, ``DIMS`` and
    ``COEFS``, elaborated with slang (once per nest and source); ``GeometryError`` when
    slang declines. FinnLib is read where it is, never fetched: without a local copy,
    ``finn.resources.ResourceError`` names it."""
    source = Path(resources.path(SOURCE.root, fetch=False)) / SOURCE.path
    return _evaluated(source, frame_words, tuple(dims), tuple(strides))


@dataclass(frozen=True)
class NestBuffer:
    """What ``input_gen``'s buffer does with one frame.

    ``capacity`` is the words it accepts beyond those it has freed (``BUF_SIZE -
    1``); per output beat of a frame, in order, ``reads`` is the input word of the
    frame it presents and ``freed`` the words of the frame freed once it is
    presented. An input word is accepted while fewer than ``capacity`` accepted
    words are unfreed; a beat is presented once the word it reads was accepted.
    """

    capacity: int
    reads: tuple[int, ...]
    freed: tuple[int, ...]


@cache
def _stepped(
    source: Path, frame_words: int, dims: tuple[int, ...], strides: tuple[int, ...]
) -> NestBuffer:
    geometry = _evaluated(source, frame_words, dims, strides)
    depth = len(dims)
    reads: list[int] = []
    freed: list[int] = []
    read = free = 0
    counters = [0] * depth
    for _ in range(prod(dims)):
        reads.append(read)
        level = depth
        while level > 0 and counters[level - 1] == dims[level - 1] - 1:
            level -= 1
        read += geometry.read_steps[level]
        if geometry.frees[level]:
            free += geometry.free_steps[level]
        freed.append(free)
        for inner in range(depth - 1, level - 1, -1):
            counters[inner] = 0
        if level > 0:
            counters[level - 1] += 1
    return NestBuffer(geometry.capacity, tuple(reads), tuple(freed))


def nest_buffer(frame_words: int, dims: Sequence[int], strides: Sequence[int]) -> NestBuffer:
    """``input_gen``'s buffer over one frame: its pointers stepped as its nest counters
    select the increments ``nest_geometry`` read from the RTL (once per nest and source).

    At each output beat, the outermost level the beat completes (with every level
    inside it) selects the read pointer's increment and, where that level frees
    (``R_FLAG``), the free pointer's; a beat completing no loop advances by the
    innermost default (level ``D``).
    """
    source = Path(resources.path(SOURCE.root, fetch=False)) / SOURCE.path
    return _stepped(source, frame_words, tuple(dims), tuple(strides))


__all__ = [
    "SOURCE",
    "Generate",
    "GeometryError",
    "InputGeneratorKernel",
    "NestBuffer",
    "NestGeometry",
    "generated",
    "input_gen_resources",
    "nest_buffer",
    "nest_geometry",
    "presented",
]
