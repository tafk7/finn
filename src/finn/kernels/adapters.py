# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream adapters: FinnLib components that change how a stream presents its tensor.

Each sits between two streams (``input_stream``, ``output_stream``) and derives
its output contract from the input's, so the adaptation it performs is the one
``classify`` names between the two forms:

- ``WidthConverterKernel`` (FinnLib ``vpc``): the same element sequence,
  ``lanes`` elements a beat instead of the input's (``WIDTH_CONVERSION``).
- ``TransposeKernel`` (FinnLib ``inner_shuffle``): a row-major ``(I, J)``
  matrix with SIMD elements of a row a beat becomes its columns, SIMD elements
  of a column a beat (``LANE_REGROUP``).

Neither emits markers, so a consumer that needs frame markers is refused. A
stream does not yet choose among adapters: that Decision needs each end to
present its own form (the D10 design, increment S1). Here they are ordinary
nodes a composite places between two of its streams.
"""

from __future__ import annotations

from math import lcm

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
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Free, Reset, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.physical.forms import TRAVERSAL, Traversal, regrouped
from finn.kernels.physical.stream import ReadyValidStream
from finn.kernels.streams import MODULE, PORT, Stream


def _clock_reset() -> tuple[Signal, Signal]:
    return (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal(
            "rst",
            Direction.IN,
            1,
            Reset(active_low=False, synchronous=True, synchronous_to=("clk",)),
        ),
    )


def _native(name: str, bits: int, endpoint: Endpoint) -> ReadyValidStream:
    side = "i" if endpoint is Endpoint.TARGET else "o"
    return ReadyValidStream(
        name, bits, endpoint, f"{side}dat", f"{side}vld", f"{side}rdy", "clk", "rst"
    )


class WidthConverterKernel(Kernel):
    """FinnLib ``vpc``: regroup a stream's elements from its lanes to ``lanes`` a beat.

    ``vpc`` converts vectors of N elements independently. N is the least common
    multiple of the two lane counts, so neither side pads; the stream must hold
    whole vectors.
    """

    id = "finnlib.vpc"
    version = "1"

    lanes: int = Param()
    input_stream: Stream = Param()
    output_stream: Stream = Param()

    @derived
    def vector(self) -> int:
        return lcm(self.input_stream.spec.form.lanes, self.lanes)

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        form, lanes = self.input_stream.spec.form, self.lanes
        elements = form.beats * form.lanes
        if lanes < 1 or elements % self.vector:
            return reject(
                "vpc-geometry",
                f"{elements} elements do not make whole {self.vector}-element vectors",
            )
        return True

    @derived(semantics=TRAVERSAL)
    def output_form(self) -> Traversal | Rejected:
        try:
            return regrouped(self.input_stream.spec.form, self.lanes)
        except ValueError as error:
            return reject("vpc-geometry", str(error))

    @view(semantics=STREAM_CONTRACT)
    def input_port(self) -> StreamContract:
        spec = self.input_stream.spec
        transport = _native("input", spec.payload_bits, Endpoint.TARGET)
        return StreamContract(transport, spec.element, spec.form, spec.repetition)

    @view(semantics=STREAM_CONTRACT)
    def output_port(self) -> StreamContract:
        spec = self.input_stream.spec
        transport = _native("output", self.lanes * spec.element.bits, Endpoint.INITIATOR)
        return StreamContract(transport, spec.element, self.output_form)

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(geometry_supported,))
    def build_requirements(self) -> ModuleBuildRequirements:
        spec = self.input_stream.spec
        parameters = (
            ("N", self.vector),
            ("PAD_ZEROS", 1),
            ("PI", spec.form.lanes),
            ("PO", self.lanes),
            ("RELAX_THROUGHPUT", 0),
            ("W", spec.element.bits),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("vpc"),
            (
                *_clock_reset(),
                *self.input_port.transport.pins(),
                *self.output_port.transport.pins(),
            ),
            tuple((name, str(value)) for name, value in parameters),
        )
        return ModuleBuildRequirements(
            WidthConverterKernel.id,
            WidthConverterKernel.version,
            parameters,
            abi,
            (CopiedSource("finnlib", "rtl/shape/vpc.sv", provides=("module:vpc",)),),
        )

    exports = {
        MODULE: build_requirements,
        PORT: {input_stream: input_port, output_stream: output_port},
    }


class TransposeKernel(Kernel):
    """FinnLib ``inner_shuffle``: an (I, J) matrix's rows in, its columns out.

    The input is the last two axes of its tensor in row-major order, SIMD
    elements of a row a beat; any outer axes are a sequence of matrices. The
    output presents each matrix column by column, SIMD elements of a column a
    beat. SIMD divides I and J.

    Known defect (FinnLib ``b9262df``): with SIMD 4 and a side of 4 or 8, the
    RTL emits undefined lanes when its input arrives in bursts with idle cycles
    between them; FinnLib's own testbench fails the same way with that input
    timing. The condition is not characterized, so nothing is refused yet.
    """

    id = "finnlib.inner_shuffle"
    version = "1"

    input_stream: Stream = Param()
    output_stream: Stream = Param()
    ram_style: str = Decision(values=("auto", "distributed", "block", "ultra"))

    @derived(semantics=default_semantics(tuple))
    def matrix(self) -> tuple[int, int, int] | Rejected:
        """(I, J, SIMD) of the input, which must be row-major with SIMD lanes along J."""
        form = self.input_stream.spec.form
        shape, simd = form.shape, form.lanes
        if len(shape) < 2 or shape[-1] % simd or shape[-2] % simd:
            return reject("transpose-form", "a matrix whose sides SIMD divides is required")
        rows, cols, last = shape[-2], shape[-1], len(shape) - 1
        expected = Traversal.over(
            shape,
            (
                *((axis, shape[axis], 1) for axis in range(last - 1)),
                (last - 1, rows, 1),
                (last, cols // simd, simd),
            ),
            ((last, simd, 1),),
        )
        if form != expected:
            return reject(
                "transpose-form", "the input must walk its matrices row-major, SIMD per beat"
            )
        return (rows, cols, simd)

    @derived(semantics=TRAVERSAL)
    def output_form(self) -> Traversal:
        form = self.input_stream.spec.form
        rows, cols, simd = self.matrix
        shape, last = form.shape, len(form.shape) - 1
        return Traversal.over(
            shape,
            (
                *((axis, shape[axis], 1) for axis in range(last - 1)),
                (last, cols, 1),
                (last - 1, rows // simd, simd),
            ),
            ((last - 1, simd, 1),),
        )

    @view(semantics=STREAM_CONTRACT)
    def input_port(self) -> StreamContract:
        spec = self.input_stream.spec
        transport = _native("input", spec.payload_bits, Endpoint.TARGET)
        return StreamContract(transport, spec.element, spec.form, spec.repetition)

    @view(semantics=STREAM_CONTRACT)
    def output_port(self) -> StreamContract:
        spec = self.input_stream.spec
        transport = _native("output", spec.payload_bits, Endpoint.INITIATOR)
        return StreamContract(transport, spec.element, self.output_form)

    @view(semantics=default_semantics(ModuleBuildRequirements))
    def build_requirements(self) -> ModuleBuildRequirements:
        spec = self.input_stream.spec
        rows, cols, simd = self.matrix
        parameters = (
            ("BITS", spec.element.bits),
            ("I", rows),
            ("J", cols),
            ("RAM_STYLE", f'"{self.ram_style}"'),
            ("SIMD", simd),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("inner_shuffle"),
            (
                *_clock_reset(),
                *self.input_port.transport.pins(),
                *self.output_port.transport.pins(),
            ),
            tuple((name, str(value)) for name, value in parameters),
        )
        return ModuleBuildRequirements(
            TransposeKernel.id,
            TransposeKernel.version,
            parameters,
            abi,
            (
                CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),
                CopiedSource(
                    "finnlib",
                    "rtl/infra/elasticmem.sv",
                    provides=("module:elasticmem",),
                    requires=("module:fifo",),
                ),
                CopiedSource(
                    "finnlib",
                    "rtl/shape/inner_shuffle.sv",
                    provides=("module:inner_shuffle",),
                    requires=("module:elasticmem", "module:fifo"),
                ),
            ),
        )

    exports = {
        MODULE: build_requirements,
        PORT: {input_stream: input_port, output_stream: output_port},
    }


__all__ = ["TransposeKernel", "WidthConverterKernel"]
