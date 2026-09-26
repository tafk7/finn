# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by extents. Zero strides repeat words. This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.
"""

from __future__ import annotations

from finn.kernels.base import Kernel
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
from finn.kernels.physical.stream import (
    STREAM_INTERFACES,
    MarkerKind,
    ReadyValidStream,
    StreamMarker,
)
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.datatypes.semantics import INTEGER_VECTOR, IntegerVector
from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    default_semantics,
    reject,
    view,
)


class InputGeneratorKernel(Kernel):
    id = "finnlib.input_generator"
    version = "1"

    word_bits: Param[int] = Param(int)
    frame_words: Param[int] = Param(int)
    extents: Param[IntegerVector] = Param(INTEGER_VECTOR)
    strides: Param[IntegerVector] = Param(INTEGER_VECTOR)

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

    ram_style = Decision(str, values=("auto", "distributed", "block", "ultra"))

    @view(semantics=STREAM_INTERFACES)
    def interfaces(self) -> tuple[ReadyValidStream, ...] | Rejected:
        bits, rank = self.word_bits, len(self.extents)
        if not 1 <= bits <= 0xFFFFFFFF or rank < 1:
            return reject(
                "input-generator-interface", "positive native word width and rank are required"
            )
        return (
            ReadyValidStream("input", bits, Endpoint.TARGET, "idat", "ivld", "irdy", "clk", "rst"),
            ReadyValidStream(
                "output",
                bits,
                Endpoint.INITIATOR,
                "odat",
                "ovld",
                "ordy",
                "clk",
                "rst",
                (StreamMarker("olst", MarkerKind.LOOP_END, rank),),
            ),
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(traversal_supported,))
    def build_requirements(self) -> ModuleBuildRequirements | Rejected:
        bits = self.word_bits
        frame = self.frame_words
        extents = self.extents
        strides = self.strides
        ram = self.ram_style
        streams = self.interfaces()
        parameters = (
            ("COEFS", "'{" + ", ".join(map(str, strides)) + "}"),
            ("D", len(extents)),
            ("DATA_WIDTH", bits),
            ("DIMS", "'{" + ", ".join(map(str, extents)) + "}"),
            ("FM_SIZE", frame),
            ("RAM_STYLE", f'"{ram}"'),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("input_gen"),
            (
                Signal("clk", Direction.IN, 1, Clock()),
                Signal(
                    "rst",
                    Direction.IN,
                    1,
                    Reset(active_low=False, synchronous=True, synchronous_to=("clk",)),
                ),
                *(pin for stream in streams for pin in stream.pins()),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        return ModuleBuildRequirements(
            InputGeneratorKernel.id,
            InputGeneratorKernel.version,
            parameters,
            abi,
            (CopiedSource("finnlib", "rtl/input_gen.sv", provides=("module:input_gen",)),),
        )


__all__ = ["InputGeneratorKernel"]
