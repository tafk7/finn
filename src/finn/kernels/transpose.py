# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``inner_shuffle`` as a node between two streams: a banked matrix transpose.

A row-major ``(I, J)`` matrix, SIMD elements of a row a beat, becomes its
columns, SIMD elements of a column a beat (``LANE_REGROUP``). It is not a
candidate of a stream's ``adapter`` Decision (``finn.kernels.adapters``): a
kernel with children places it explicitly, and a stream realizes lane regroups
through the common lane count instead. The defect that kept it out (see
``TransposeKernel``) is fixed in FinnLib, which reopens that option.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    derived,
    divisors_of,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Index, Schedule
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel, extent_of
from finn.kernels.channels import Channel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import AxiStreamPort

i, j = Index("i"), Index("j")


class TransposeKernel(Kernel):
    """FinnLib ``inner_shuffle``: an (I, J) matrix's rows in, its columns out.

    The input is the last two axes of its tensor in row-major order, SIMD
    elements of a row a beat; any outer axes are a sequence of matrices. The
    output presents each matrix column by column, SIMD elements of a column a
    beat. SIMD, a Decision, divides I and J; I and J are bound from the ports.

    Needs FinnLib ``99d75e8`` ("inner_shuffle: read a page only once all of it
    is written"), which is pinned; ``d03f2fc`` predates it. The RTL writes matrices
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
    version = 1
    rtl_module = "inner_shuffle"

    input_stream: Channel = Param(required=False)
    output_stream: Channel = Param(required=False)
    ram_style: str = Decision(values=("auto", "distributed", "block", "ultra"))

    rows = extent_of(i)  # I
    cols = extent_of(j)  # J

    @derived
    def sides(self) -> int:
        """The common divisors of I and J are SIMD's domain."""
        return gcd(self.rows, self.cols)

    simd: int = Decision(domain=divisors_of(sides))

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes: any leading ones (a sequence of matrices), then ``i`` and ``j``."""
        rank = len(self.input_stream.tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 2)), i, j)

    @derived
    def rows_in(self) -> Schedule | Rejected:
        """Each matrix's rows in turn, SIMD elements of a row a beat."""
        return self.bound_schedule(self.indices, {j: self.simd})

    @derived
    def columns_out(self) -> Schedule | Rejected:
        """Each matrix's columns in turn, SIMD elements of a column a beat."""
        *outer, _, _ = self.indices
        return self.bound_schedule((*outer, j, i), {i: self.simd})

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def dtype(self) -> QONNXDataType:
        """The element it moves: the input's, unchanged."""
        return self.input_stream.tensor.element.dtype

    input = AxiStreamPort(
        name="input",
        endpoint=Endpoint.TARGET,
        stream=input_stream,
        schedule=rows_in,
        index=indices,
        lanes=(j,),
        dtype=dtype,
        signals=("idat", "ivld", "irdy"),
        clock="clk",
        reset="rst",
    )
    output = AxiStreamPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        schedule=columns_out,
        index=indices,
        lanes=(i,),
        dtype=dtype,
        signals=("odat", "ovld", "ordy"),
        clock="clk",
        reset="rst",
    )

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "BITS": self.input_stream.tensor.element.bits,
            "I": self.rows,
            "J": self.cols,
            "RAM_STYLE": f'"{self.ram_style}"',
            "SIMD": self.simd,
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
