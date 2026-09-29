# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by extents. Zero strides repeat words. This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.

On a stream, ``input_gen`` is a stage of the stream's adapter
(``finn.kernels.adapters``), which derives these facts from the stream's plan.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import CLOCKING, NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.datatypes.semantics import INTEGER_VECTOR, IntegerVector
from finn.kernels.physical.stream import (
    STREAM_INTERFACES,
    MarkerKind,
    ReadyValidStream,
    StreamMarker,
)
from finn.kernels.port import MARKERS, WordPort

INPUT_GEN_RAM_STYLES = ("auto", "distributed", "block", "ultra")


def _vector(values: IntegerVector) -> str:
    return "'{" + ", ".join(map(str, values)) + "}"


class InputGeneratorKernel(Kernel):
    id = "finnlib.input_generator"
    version = "1"
    module = "input_gen"

    word_bits: int = Param()
    frame_words: int = Param()
    extents: IntegerVector = Param(semantics=INTEGER_VECTOR)
    strides: IntegerVector = Param(semantics=INTEGER_VECTOR)

    @constraint
    def traversal_supported(self) -> bool | Rejected:
        bits = self.word_bits
        frame = self.frame_words
        extents = self.extents
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
    ram_style: str = Decision(values=INPUT_GEN_RAM_STYLES)

    @derived(semantics=MARKERS)
    def loop_ends(self) -> tuple[StreamMarker, ...] | Rejected:
        """``olst``: one bit per loop, closing that loop and every inner one."""
        rank = len(self.extents)
        if rank < 1:
            return reject("input-generator-interface", "a traversal has at least one loop")
        return (StreamMarker("olst", MarkerKind.LOOP_END, rank),)

    input = WordPort(name="input", endpoint=Endpoint.TARGET, bits=word_bits)
    output = WordPort(name="output", endpoint=Endpoint.INITIATOR, bits=word_bits, markers=loop_ends)

    @view(semantics=STREAM_INTERFACES)
    def interfaces(self) -> tuple[ReadyValidStream, ...] | Rejected:
        return (self.input.transport, self.output.transport)

    @derived(semantics=CLOCKING)
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "COEFS": _vector(self.strides),
            "D": len(self.extents),
            "DATA_WIDTH": self.word_bits,
            "DIMS": _vector(self.extents),
            "FM_SIZE": self.frame_words,
            "RAM_STYLE": f'"{self.ram_style}"',
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",)),)


__all__ = ["INPUT_GEN_RAM_STYLES", "InputGeneratorKernel"]
