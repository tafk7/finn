# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``inner_shuffle`` as a node from one tensor to another: a banked matrix transpose.

A row-major ``(I, J)`` matrix, SIMD elements of it a beat, becomes its
columns, SIMD elements of a column a beat (``LANE_REGROUP``). It is not a
candidate of a channel's ``adapter`` Decision (``finn.kernels.adapters``): a
kernel with children places it explicitly, and a channel realizes lane regroups
through the common lane count instead.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    reject,
    view,
)
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace, extent_of
from finn.kernels.fifo import fifo_resources
from finn.kernels.library import LibraryModule
from finn.kernels.port import AxiStreamPort
from finn.kernels.utilization import (
    RESOURCES_SEMANTICS,
    auto_unstated,
    memory,
    memory_styles,
    uram_constraint,
)
from finn.platform.capabilities import Fabric, Platform, Resources

i, j = Index("i"), Index("j")


def inner_shuffle_resources(
    *, bits: int, i: int, j: int, simd: int, ram_style: str, fabric: Fabric
) -> Resources:
    """FinnLib ``inner_shuffle``, from its RTL's structure (not characterised): SIMD
    banks of two pages, BANK_DEPTH = 2 I J / SIMD elements each, simple dual port, in
    RAM_STYLE on ``fabric``; the rotators on both sides, a SIMD-way multiplexer a lane
    bit, and a register a lane bit at each; its two skid FIFOs of depth two (FinnLib
    ``fifo``), the output's and the read pattern's."""
    index_bits = max(simd - 1, 0).bit_length()
    banks = memory(2 * i * j // simd, bits, ram_style, fabric=fabric).times(simd)
    rotators = Resources(lut=2 * simd * bits * index_bits, ff=2 * simd * bits)
    skids = fifo_resources(2, simd * bits, "auto", fabric=fabric)
    if index_bits:
        skids = skids + fifo_resources(2, simd * index_bits, "auto", fabric=fabric)
    return banks + rotators + skids


class TransposeKernel(KernelSpace):
    """FinnLib ``inner_shuffle``: an (I, J) matrix's rows in, its columns out.

    The input is the last two axes of its tensor in row-major order, SIMD
    consecutive elements a beat; any outer axes are a sequence of matrices. The
    output presents each matrix column by column, SIMD elements of a column a
    beat. SIMD, a Decision, divides I, the RTL's one constraint
    (``inner_shuffle.sv``: ``I % SIMD == 0``); I and J are the input's last axes.

    The RTL writes its input to consecutive bank addresses, so a beat may span
    two rows when SIMD does not divide J (it rotates its write banks by
    ``gcd(J, SIMD)``). The input reads its matrix through a row-major ``(J, I)``
    view: the same flat order, walked by the output's schedule, ``I / SIMD``
    beats of SIMD lanes for each of J. A beat never spans two matrices. Its output,
    the tensor it states (``output_tensor``), is its input's shape and element, its
    data presented column by column. Its cycles are the beats its ports carry, as
    every leaf's.

    The RTL writes matrices alternately into two pages, and reads a page only
    once all of it is written, its last beat included (FinnLib ``99d75e8`` and
    later; the pin includes it). Without that guard an output draining faster
    than the input arrives (stalled or bursty input, as behind a ``vpc``) reads
    lanes not yet written. ``tests/support/kernel_specs/transpose.py``'s ``transpose``
    case runs stalled (and a ``vpc`` feeds its adapter sample), so it checks the
    guard.

    The pages' ``ram_style`` is its choice: the explicit styles, ordered by the bits both
    pages hold (``pages_bits``), then ``auto``, which states no resources
    (``finn.kernels.utilization.memory_styles``); ``ultra`` needs the ``platform``'s
    UltraRAM, a constraint of the kernel (``uram-absent``; the pages start empty, so no
    initial contents are asked of it). The RTL's own limit is a constraint too: it counts
    its banks' two pages, ``2 I J`` elements, in 32 bits. A port reading fewer than two
    axes refuses itself.
    """

    id = "finnlib.inner_shuffle"
    clock_domain = NATIVE_CLOCKING

    # The tensor its input port reads: a fact it is given.
    input_tensor: Tensor = Param(required=False)
    platform: Platform = Param()

    @derived
    def extents(self) -> dict[Index, int] | Rejected:
        """Each index's extent: the input reads its tensor through a view, which binds
        none, so the kernel gives its axes' extents."""
        return self._bound(dict(zip(self.indices, self.input_tensor.shape)))

    rows = extent_of(i)  # I
    cols = extent_of(j)  # J

    @derived
    def pages_bits(self) -> int:
        """The bits both pages hold: ``2 I J`` elements."""
        return 2 * self.rows * self.cols * self.input_tensor.element.bits

    ram_style: str = Decision(domain=memory_styles(pages_bits))
    uram_present = uram_constraint(platform, ram_style, ("ultra",))

    simd: int = Decision(domain=divisors_of(rows))

    @constraint
    def pages_supported(self) -> bool | Rejected:
        """The RTL counts its banks' two pages, ``2 I J`` elements, in 32 bits."""
        if 2 * self.rows * self.cols > 0xFFFFFFFF:
            return reject("transpose-depth", "two pages of I x J elements overflow 32 bits")
        return True

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes: any leading ones (a sequence of matrices), then ``i`` and ``j``."""
        rank = len(self.input_tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 2)), i, j)

    @derived
    def columns(self) -> tuple[Index, ...]:
        """The output's walk: any outer axes, then ``j``, then ``i``."""
        *outer, _, _ = self.indices
        return (*outer, j, i)

    @derived
    def columns_out(self) -> Schedule | Rejected:
        """Each matrix's columns in turn, SIMD elements of a column a beat. The input,
        read as each matrix's ``(J, I)`` view, walks it too: its rows, flat."""
        return self.bound_schedule(self.columns, {i: self.simd})

    @derived
    def output_tensor(self) -> Tensor | Rejected:
        """Its output: the input's shape and element, unchanged."""
        shape = self.spanned(self.indices)
        if isinstance(shape, Rejected):
            return shape
        return Tensor(shape, self.input_tensor.element)

    input = AxiStreamPort(
        name="input",
        endpoint=Endpoint.TARGET,
        tensor=input_tensor,
        schedule=columns_out,
        index=columns,
        lanes=(i,),
        reshaped=True,
        signals=("idat", "ivld", "irdy"),
    )
    output = AxiStreamPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        tensor=output_tensor,
        schedule=columns_out,
        index=indices,
        lanes=(i,),
        signals=("odat", "ovld", "ordy"),
    )

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        if self.ram_style == "auto":
            return auto_unstated("the transpose's pages")
        return inner_shuffle_resources(
            bits=self.input_tensor.element.bits,
            i=self.rows,
            j=self.cols,
            simd=self.simd,
            ram_style=self.ram_style,
            fabric=self.platform.fabric,
        )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "BITS": self.input_tensor.element.bits,
            "I": self.rows,
            "J": self.cols,
            "RAM_STYLE": f'"{self.ram_style}"',
            "SIMD": self.simd,
        }

    module = LibraryModule(
        "inner_shuffle",
        parameters=parameters,
        sources=(
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


__all__ = ["TransposeKernel", "inner_shuffle_resources"]
