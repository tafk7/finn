# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by its loop extents (``dims``). Zero strides repeat words.
This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.

On a channel, ``input_gen`` is a stage of the channel's adapter
(``finn.kernels.adapters``), which derives these facts from the channel's plan.
Its buffer's ``ram_style`` is its choice; ``ultra`` requires the ``platform``'s
UltraRAM. The buffer starts empty, so no initial contents are asked of it.

What the buffer does is read from the RTL, not copied (decision FS6):
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from math import prod
from pathlib import Path

from finn import resources
from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    requires,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.rtl import Declined, evaluate
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.port import WordPort
from finn.kernels.target import Platform
from finn.kernels.transport import MarkerKind, StreamMarker
from finn.kernels.utilization import RESOURCES_SEMANTICS, Fit, Resources, memory
from finn.kernels.values.semantics import INTEGER_VECTOR, IntegerVector

_INPUT_GEN_RAM_STYLES = ("auto", "distributed", "block", "ultra")
SOURCE = CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",))


def _vector(values: IntegerVector) -> str:
    return "'{" + ", ".join(map(str, values)) + "}"


def input_gen_resources(*, buf_size: int, data_width: int, d: int, ram_style: str) -> Resources:
    """FinnLib ``input_gen``: its buffer, BUF_SIZE words of DATA_WIDTH bits, simple dual
    port, in RAM_STYLE (``nest_geometry`` reads BUF_SIZE from the RTL), and its read and
    output registers, two a bit. The nest's D counters and pointers are a ``Fit`` over
    their address bits."""
    address = d * max(buf_size - 1, 1).bit_length()
    return memory(buf_size, data_width, ram_style) + Resources(
        lut=_INPUT_GEN_LUT.at(address), ff=2 * data_width + _INPUT_GEN_FF.at(address)
    )


# Feature: the loops' address bits, D * clog2(BUF_SIZE).
_INPUT_GEN_LUT = Fit(9.9, (3.07,))
_INPUT_GEN_FF = Fit(12.7, (2.31,))


class InputGeneratorKernel(Kernel):
    id = "finnlib.input_generator"
    version = 1
    rtl_module = "input_gen"

    word_bits: int = Param()
    frame_words: int = Param()
    dims: IntegerVector = Param(semantics=INTEGER_VECTOR)
    strides: IntegerVector = Param(semantics=INTEGER_VECTOR)
    platform: Platform = Param()

    @constraint
    def traversal_supported(self) -> bool | Rejected:
        bits = self.word_bits
        frame = self.frame_words
        extents = self.dims
        strides = self.strides
        if bits < 1 or frame < 1 or not extents or len(extents) != len(strides):
            return reject(
                "input-generator-shape",
                "positive word/frame sizes and equally ranked nonempty vectors are required",
            )
        if any(extent < 1 for extent in extents) or any(stride < 0 for stride in strides):
            return reject(
                "input-generator-loop", "loop extents are positive and strides nonnegative"
            )
        if bits > 0xFFFFFFFF or any(value > 0x7FFFFFFF for value in (frame, *extents, *strides)):
            return reject(
                "input-generator-range", "loop arithmetic must fit native signed 32-bit increments"
            )
        if sum((extent - 1) * stride for extent, stride in zip(extents, strides)) >= frame:
            return reject(
                "input-generator-address", "every selected word must lie within its input frame"
            )
        return True

    admission = ConstraintGroup(traversal_supported)
    ram_style: str = Decision(
        values=_INPUT_GEN_RAM_STYLES,
        requires=(
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=("ultra",)),
        ),
    )

    @derived
    def loop_ends(self) -> tuple[StreamMarker, ...] | Rejected:
        """``olst``: one bit per loop, closing that loop and every inner one."""
        rank = len(self.dims)
        if rank < 1:
            return reject("input-generator-interface", "a traversal has at least one loop")
        return (StreamMarker("olst", MarkerKind.LOOP_END, rank),)

    input = WordPort(name="input", endpoint=Endpoint.TARGET, bits=word_bits)
    output = WordPort(name="output", endpoint=Endpoint.INITIATOR, bits=word_bits, markers=loop_ends)

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        try:
            words = nest_geometry(self.frame_words, self.dims, self.strides).buffer_words
        except GeometryError as error:
            return reject("input-generator-geometry", str(error))
        return input_gen_resources(
            buf_size=words, data_width=self.word_bits, d=len(self.dims), ram_style=self.ram_style
        )

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "COEFS": _vector(self.strides),
            "D": len(self.dims),
            "DATA_WIDTH": self.word_bits,
            "DIMS": _vector(self.dims),
            "FM_SIZE": self.frame_words,
            "RAM_STYLE": f'"{self.ram_style}"',
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (SOURCE,)


# -- the buffer, as the RTL derives it -------------------------------------------------------


class GeometryError(ValueError):
    """A nest whose buffer could not be read from the RTL: slang declined, named."""


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


def nest_buffer(frame_words: int, dims: Sequence[int], strides: Sequence[int]) -> NestBuffer:
    """``input_gen``'s buffer over one frame: its pointers stepped as its nest counters
    select the increments ``nest_geometry`` read from the RTL.

    At each output beat, the outermost level the beat completes (with every level
    inside it) selects the read pointer's increment and, where that level frees
    (``R_FLAG``), the free pointer's; a beat completing no loop advances by the
    innermost default (level ``D``).
    """
    geometry = nest_geometry(frame_words, dims, strides)
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


__all__ = [
    "SOURCE",
    "GeometryError",
    "InputGeneratorKernel",
    "NestBuffer",
    "NestGeometry",
    "input_gen_resources",
    "nest_buffer",
    "nest_geometry",
]
