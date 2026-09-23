# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by extents. Zero strides repeat words. This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.
"""

from finn.kernels._engine import ValueSemantics
from finn.kernels.artifacts.abi import Clock, Direction, Reset, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.kernels.space import (
    ConstraintGroup,
    Decision,
    Input,
    Readiness,
    View,
    constraint,
    derived,
    reject,
)

IntegerVector = tuple[int, ...]
INTEGER_VECTOR: ValueSemantics[IntegerVector] = ValueSemantics(
    IntegerVector,
    "integer vector",
    lambda value: type(value) is tuple and all(type(item) is int for item in value),
    lambda left, right: left == right,
    lambda value: value,
)


class InputGeneratorKernel(Kernel):
    id = "finnlib.input_generator"
    version = "1"

    word_bits = Input(int)
    frame_words = Input(int)
    extents = Input(INTEGER_VECTOR)
    strides = Input(INTEGER_VECTOR)
    ram_style = Decision(str, values=("auto", "distributed", "block", "ultra"))

    @constraint(bits=word_bits, frame=frame_words, extents=extents, strides=strides)
    def traversal_supported(
        *, bits: int, frame: int, extents: IntegerVector, strides: IntegerVector
    ) -> object:
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

    @derived(
        ModuleBuildRequirements,
        bits=word_bits,
        frame=frame_words,
        extents=extents,
        strides=strides,
        ram=ram_style,
    )
    def codegen(
        *, bits: int, frame: int, extents: IntegerVector, strides: IntegerVector, ram: str
    ) -> object:
        if bits < 1 or not extents:
            return reject(
                "input-generator-interface", "positive word width and nonempty extents are required"
            )
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
                Signal("idat", Direction.IN, bits),
                Signal("ivld", Direction.IN, 1),
                Signal("irdy", Direction.OUT, 1),
                Signal("odat", Direction.OUT, bits),
                Signal("ovld", Direction.OUT, 1),
                Signal("olst", Direction.OUT, len(extents)),
                Signal("ordy", Direction.IN, 1),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        return ModuleBuildRequirements(
            InputGeneratorKernel.id,
            InputGeneratorKernel.version,
            parameters,
            abi,
            (CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",)),),
        )

    support = ConstraintGroup(traversal_supported)
    physical_ready = Readiness()
    physical = View(codegen, readiness=physical_ready, constraints=support)


__all__ = ["InputGeneratorKernel"]
