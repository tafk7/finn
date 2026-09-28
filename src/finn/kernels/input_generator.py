# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by extents. Zero strides repeat words. This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.

This is the flat module. On a stream, ``input_gen`` is a stage of the stream's
adapter (``finn.kernels.adapters``), which derives these parameters from the
stream's plan.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    default_semantics,
    reject,
    view,
)
from finn.kernels.adapters import INPUT_GEN_RAM_STYLES, input_gen_interfaces, input_gen_requirements
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import INTEGER_VECTOR, IntegerVector
from finn.kernels.physical.stream import STREAM_INTERFACES, ReadyValidStream
from finn.kernels.streams import MODULE


class InputGeneratorKernel(Kernel):
    id = "finnlib.input_generator"
    version = "1"

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

    ram_style: str = Decision(values=INPUT_GEN_RAM_STYLES)

    @view(semantics=STREAM_INTERFACES)
    def interfaces(self) -> tuple[ReadyValidStream, ...] | Rejected:
        bits, rank = self.word_bits, len(self.extents)
        if not 1 <= bits <= 0xFFFFFFFF or rank < 1:
            return reject(
                "input-generator-interface", "positive native word width and rank are required"
            )
        return input_gen_interfaces(bits, rank)

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(traversal_supported,))
    def build_requirements(self) -> ModuleBuildRequirements:
        _ = self.interfaces  # refuses a word width or rank no pins have
        return input_gen_requirements(
            word_bits=self.word_bits,
            frame=self.frame_words,
            dims=self.extents,
            coefs=self.strides,
            ram_style=self.ram_style,
        )

    exports = {MODULE: build_requirements}


__all__ = ["InputGeneratorKernel"]
