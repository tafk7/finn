# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered, same-word-width traversal of consecutive input frames.

For each frame, emit the word at sum(index[i] * strides[i]) in the nested
loop order described by extents. Zero strides repeat words. This declaration
admits finite traversals wholly within each frame. olst[i] marks completion
of loop i and all inner loops, aligned with the output transfer. It is a native
multi-bit marker, not AXI TLAST. Input and output words are opaque bits.

Placed between two streams (``input_stream``, ``output_stream``), its output
contract is derived from the input's: the frames' beats are one run, and each
loop of the nest steps ``stride`` beats of it, so a replay (stride 0) or a
reorder of the frame is an ordinary loop nest. Each ``olst[i]`` is a marker bit
closing every ``extents[i] * ... * extents[-1]`` output beats.
"""

from __future__ import annotations

from math import prod

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
    derived,
    reject,
    view,
)
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.dataflow.traversal import TRAVERSAL, LevelEnd, Loop, Traversal, split_beats
from finn.kernels.streams import MODULE, PORT, Stream


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

    ram_style: str = Decision(values=("auto", "distributed", "block", "ultra"))
    # The streams it sits on, when a parent places it between streams, and the
    # form its input presents.
    input_stream: Stream = Param(required=False)
    output_stream: Stream = Param(required=False)
    input_form: Traversal = Param(semantics=TRAVERSAL, required=False)

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
        streams = self.interfaces
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
            (CopiedSource("finnlib", "rtl/shape/input_gen.sv", provides=("module:input_gen",)),),
        )

    @derived(semantics=default_semantics(Traversal))
    def output_form(self) -> Traversal | Rejected:
        """The input's frames, each presented through the loop nest."""
        form, frame = self.input_form, self.frame_words
        if form.beats % frame:
            return reject("input-generator-frame", f"{frame}-beat frames do not divide the stream")
        try:
            outer, inner = split_beats(form, frame)
        except ValueError as error:
            return reject("input-generator-frame", str(error))
        if len(inner) > 1:
            return reject("input-generator-frame", "a frame's beats must be one run of the stream")
        unit = inner[0].stride if inner else 0
        nest = tuple(
            Loop(extent, stride * unit) for extent, stride in zip(self.extents, self.strides)
        )
        try:
            return Traversal(form.shape, (*outer, *nest), form.lane_loops)
        except ValueError as error:
            return reject("input-generator-frame", str(error))

    @view(semantics=STREAM_CONTRACT)
    def input_port(self) -> StreamContract | Rejected:
        element, form = self.input_stream.tensor.element, self.input_form
        if form.lanes * element.bits != self.word_bits:
            return reject(
                "input-generator-word", "the stream's beats are not the generator's words"
            )
        return StreamContract(self.interfaces[0], element, form)

    @view(semantics=STREAM_CONTRACT)
    def output_port(self) -> StreamContract:
        element, extents = self.input_stream.tensor.element, self.extents
        markers = {
            f"olst[{level}]": LevelEnd(prod(extents[level:])) for level in range(len(extents))
        }
        return StreamContract(self.interfaces[1], element, self.output_form, markers=markers)

    exports = {
        MODULE: build_requirements,
        PORT: {input_stream: input_port, output_stream: output_port},
    }


__all__ = ["InputGeneratorKernel"]
