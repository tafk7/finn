# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``inner_shuffle`` as a node between two streams: a banked matrix transpose.

A row-major ``(I, J)`` matrix, SIMD elements of a row a beat, becomes its
columns, SIMD elements of a column a beat (``LANE_REGROUP``). It is not yet a
candidate of a stream's ``adapter`` Decision (``finn.kernels.adapters``): under
bursty input it emits undefined lanes (see ``TransposeKernel``), so a composite
places it explicitly, and a stream realizes lane regroups through the common
lane count instead.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.dataflow.traversal import TRAVERSAL, Traversal
from finn.kernels.adapters import clock_reset, native
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.streams import MODULE, PORT, Stream


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
    input_form: Traversal = Param(semantics=TRAVERSAL)
    ram_style: str = Decision(values=("auto", "distributed", "block", "ultra"))

    @derived(semantics=default_semantics(tuple))
    def matrix(self) -> tuple[int, int, int] | Rejected:
        """(I, J, SIMD) of the input, which must be row-major with SIMD lanes along J."""
        form = self.input_form
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
        form = self.input_form
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
        element, form = self.input_stream.tensor.element, self.input_form
        transport = native("input", form.lanes * element.bits, Endpoint.TARGET)
        return StreamContract(transport, element, form)

    @view(semantics=STREAM_CONTRACT)
    def output_port(self) -> StreamContract:
        element, form = self.input_stream.tensor.element, self.input_form
        transport = native("output", form.lanes * element.bits, Endpoint.INITIATOR)
        return StreamContract(transport, element, self.output_form)

    @view(semantics=default_semantics(ModuleBuildRequirements))
    def build_requirements(self) -> ModuleBuildRequirements:
        element = self.input_stream.tensor.element
        rows, cols, simd = self.matrix
        parameters = (
            ("BITS", element.bits),
            ("I", rows),
            ("J", cols),
            ("RAM_STYLE", f'"{self.ram_style}"'),
            ("SIMD", simd),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("inner_shuffle"),
            (
                *clock_reset(),
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


__all__ = ["TransposeKernel"]
