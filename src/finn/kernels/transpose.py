# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``inner_shuffle`` as a node between two channels: a banked matrix transpose.

A row-major ``(I, J)`` matrix, SIMD elements of a row a beat, becomes its
columns, SIMD elements of a column a beat (``LANE_REGROUP``). It is not a
candidate of a channel's ``adapter`` Decision (``finn.kernels.adapters``): a
kernel with children places it explicitly, and a channel realizes lane regroups
through the common lane count instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    reject,
    requires,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Index, Schedule
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel, extent_of
from finn.kernels.channels import Channel
from finn.kernels.fifo import fifo_resources
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import Platform
from finn.kernels.utilization import RESOURCES_SEMANTICS, Resources, memory
from finn.kernels.values.semantics import QONNX_DATATYPE_VALUE_SEMANTICS

i, j = Index("i"), Index("j")


def inner_shuffle_resources(*, bits: int, i: int, j: int, simd: int, ram_style: str) -> Resources:
    """FinnLib ``inner_shuffle``, from its RTL's structure (not characterised): SIMD
    banks of two pages, BANK_DEPTH = 2 I J / SIMD elements each, simple dual port, in
    RAM_STYLE; the rotators on both sides, a SIMD-way multiplexer a lane bit, and a
    register a lane bit at each; its two skid FIFOs of depth two (FinnLib ``fifo``), the
    output's and the read pattern's."""
    index_bits = max(simd - 1, 0).bit_length()
    banks = memory(2 * i * j // simd, bits, ram_style).times(simd)
    rotators = Resources(lut=2 * simd * bits * index_bits, ff=2 * simd * bits)
    skids = fifo_resources(2, simd * bits, "auto")
    if index_bits:
        skids = skids + fifo_resources(2, simd * index_bits, "auto")
    return banks + rotators + skids


class TransposeKernel(Kernel):
    """FinnLib ``inner_shuffle``: an (I, J) matrix's rows in, its columns out.

    The input is the last two axes of its tensor in row-major order, SIMD
    elements of a row a beat; any outer axes are a sequence of matrices. The
    output presents each matrix column by column, SIMD elements of a column a
    beat. SIMD, a Decision, divides I and J; I and J are bound from the ports.

    The RTL writes matrices alternately into two pages, and reads a page only
    once all of it is written, its last beat included (FinnLib ``99d75e8`` and
    later; the pin includes it). Without that guard an output draining faster
    than the input arrives (stalled or bursty input, as behind a ``vpc``) reads
    lanes not yet written. ``tests/kernels/test_conformance.py``'s transpose
    case runs stalled and behind a ``vpc``, so it checks the guard.

    The pages' ``ram_style`` is its choice; ``ultra`` requires the ``platform``'s
    UltraRAM (the pages start empty, so no initial contents are asked of it).
    Admission is the RTL's own limit: it counts its banks' two pages, ``2 I J``
    elements, in 32 bits. A port reading fewer than two axes refuses itself.
    """

    id = "finnlib.inner_shuffle"
    version = 1
    rtl_module = "inner_shuffle"

    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    platform: Platform = Param()
    ram_style: str = Decision(
        values=("auto", "distributed", "block", "ultra"),
        requires=(
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=("ultra",)),
        ),
    )

    rows = extent_of(i)  # I
    cols = extent_of(j)  # J

    @derived
    def sides(self) -> int:
        """The common divisors of I and J are SIMD's domain."""
        return gcd(self.rows, self.cols)

    simd: int = Decision(domain=divisors_of(sides))

    @constraint
    def pages_supported(self) -> bool | Rejected:
        """The RTL counts its banks' two pages, ``2 I J`` elements, in 32 bits."""
        if 2 * self.rows * self.cols > 0xFFFFFFFF:
            return reject("transpose-depth", "two pages of I x J elements overflow 32 bits")
        return True

    admission = ConstraintGroup(pages_supported)

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes: any leading ones (a sequence of matrices), then ``i`` and ``j``."""
        rank = len(self.input_channel.tensor.shape)
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
        return self.input_channel.tensor.element.dtype

    input = AxiStreamPort(
        name="input",
        endpoint=Endpoint.TARGET,
        channel=input_channel,
        schedule=rows_in,
        index=indices,
        lanes=(j,),
        dtype=dtype,
        signals=("idat", "ivld", "irdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    output = AxiStreamPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        channel=output_channel,
        schedule=columns_out,
        index=indices,
        lanes=(i,),
        dtype=dtype,
        signals=("odat", "ovld", "ordy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return inner_shuffle_resources(
            bits=self.input_channel.tensor.element.bits,
            i=self.rows,
            j=self.cols,
            simd=self.simd,
            ram_style=self.ram_style,
        )

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "BITS": self.input_channel.tensor.element.bits,
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


__all__ = ["TransposeKernel", "inner_shuffle_resources"]
