# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``inner_shuffle`` as a node between two streams: a banked matrix transpose.

A row-major ``(I, J)`` matrix, SIMD elements of a row a beat, becomes its
columns, SIMD elements of a column a beat (``LANE_REGROUP``). It is not a
candidate of a stream's ``adapter`` Decision (``finn.kernels.adapters``): a
composite places it explicitly, and a stream realizes lane regroups through the
common lane count instead. The defect that kept it out (see
``TransposeKernel``) is fixed in FinnLib, which reopens that option.
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
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.traversal import (
    BeatSequence,
    Traversal,
    vector_major,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import GivenPort
from finn.kernels.streams import Stream


class TransposeKernel(Kernel):
    """FinnLib ``inner_shuffle``: an (I, J) matrix's rows in, its columns out.

    The input is the last two axes of its tensor in row-major order, SIMD
    elements of a row a beat; any outer axes are a sequence of matrices. The
    output presents each matrix column by column, SIMD elements of a column a
    beat. SIMD divides I and J.

    Needs FinnLib ``99d75e8`` ("inner_shuffle: read a page only once all of it
    is written"); the pinned ``d03f2fc`` predates it. The RTL writes matrices
    alternately into two pages. Before the fix its read guard held the reader
    back from the first page only, so whenever the output drained faster than
    the input arrived (stalled or bursty input, as behind a ``vpc``) it read the
    second page before that page's last rows were written: undefined lanes, the
    lanes of the last rows. And a page counted as written once the write address
    reached its last beat, before that beat was written, so an input pausing
    there released the page early and then replayed it. Every SIMD (1 included)
    and shape tried failed under a slow enough input; a free-running input
    never failed. ``tests/kernels/test_conformance.py``'s transpose case fails
    against ``d03f2fc``.
    """

    id = "finnlib.inner_shuffle"
    version = "1"
    module = "inner_shuffle"

    input_stream: Stream = Param()
    output_stream: Stream = Param()
    input_form: Traversal = Param()
    ram_style: str = Decision(values=("auto", "distributed", "block", "ultra"))

    @derived
    def matrix(self) -> tuple[int, int, int] | Rejected:
        """(I, J, SIMD) of the input, which must be row-major with SIMD lanes along J."""
        form = self.input_form
        shape, simd = form.shape, form.lanes
        if len(shape) < 2 or shape[-1] % simd or shape[-2] % simd:
            return reject("transpose-form", "a matrix whose sides SIMD divides is required")
        if form != vector_major(shape, simd):
            return reject(
                "transpose-form", "the input must walk its matrices row-major, SIMD per beat"
            )
        return (shape[-2], shape[-1], simd)

    @constraint
    def transposable(self) -> bool | Rejected:
        """A banked matrix: row-major, SIMD dividing both sides."""
        _ = self.matrix
        return True

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def dtype(self) -> QONNXDataType:
        """The element it moves: the input's, unchanged."""
        return self.input_stream.tensor.element.dtype

    admission = ConstraintGroup(transposable)

    @derived
    def input_sequence(self) -> BeatSequence:
        return BeatSequence(self.input_form)

    @derived
    def output_sequence(self) -> BeatSequence:
        form = self.input_form
        rows, cols, simd = self.matrix
        shape, last = form.shape, len(form.shape) - 1
        return BeatSequence(
            Traversal.over(
                shape,
                (
                    *((axis, shape[axis], 1) for axis in range(last - 1)),
                    (last, cols, 1),
                    (last - 1, rows // simd, simd),
                ),
                ((last - 1, simd, 1),),
            )
        )

    input = GivenPort(
        name="input",
        endpoint=Endpoint.TARGET,
        stream=input_stream,
        sequence=input_sequence,
        dtype=dtype,
        signals=("idat", "ivld", "irdy"),
        clock="clk",
        reset="rst",
    )
    output = GivenPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        sequence=output_sequence,
        dtype=dtype,
        signals=("odat", "ovld", "ordy"),
        clock="clk",
        reset="rst",
    )

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        rows, cols, simd = self.matrix
        return {
            "BITS": self.input_stream.tensor.element.bits,
            "I": rows,
            "J": cols,
            "RAM_STYLE": f'"{self.ram_style}"',
            "SIMD": simd,
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (
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
        )


__all__ = ["TransposeKernel"]
