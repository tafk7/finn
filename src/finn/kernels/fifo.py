# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's opaque-word FIFO, with its native unpadded ready/valid pins.

Words have no numerical datatype. DEPTH is the requested capacity; the native
implementation may round its storage up and forces a shift FIFO for shallow
depths. ``auto`` selects by depth and word width: a shift register up to 64
words narrower than 12 bits, LUTRAM up to 257 words, then block and UltraRAM.
Reset is synchronous, active-high, and discards pending words. ``ultra``
requires the ``platform``'s UltraRAM; the FIFO starts empty, so no initial
contents are asked of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    requires,
    view,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.port import WordPort
from finn.kernels.target import Platform


@dataclass(frozen=True)
class FifoStorage:
    """The storage the native RTL selects for ``ram_style`` and DEPTH, and the words it
    accepts; synthesis resource inference is separate."""

    effective_style: str
    capacity: int


class FifoKernel(Kernel):
    id = "finnlib.fifo"
    version = 2
    rtl_module = "fifo"

    word_bits: int = Param()
    depth: int = Param()
    platform: Platform = Param()

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        bits = self.word_bits
        depth = self.depth
        if bits < 1 or depth < 2 or max(bits, depth) > 0xFFFFFFFF:
            return reject("fifo-geometry", "word_bits must be positive and depth at least two")
        return True

    ram_style: str = Decision(
        values=("auto", "shift", "distributed", "block", "ultra"),
        requires=(
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=("ultra",)),
        ),
    )

    @view(requires=(geometry_supported,))
    def storage(self) -> FifoStorage | Rejected:
        depth, style, bits = self.depth, self.ram_style, self.word_bits
        effective = (
            "shift"
            if depth <= 33
            else style
            if style != "auto"
            else "shift"
            if depth <= 64 and bits < 12
            else "distributed"
            if depth <= 257
            else "block"
            if depth <= 2028
            else "ultra"
        )
        if effective == "shift":
            capacity = max(5, depth)
        elif effective == "distributed":
            # DEPTH - 1 LUTRAM entries behind one output register.
            capacity = depth
        else:
            # Native memory decomposition; include the BRAM read pipeline or
            # the URAM credit-limited output queue in accepted-word capacity.
            ultra = effective == "ultra"
            required = depth - (17 if ultra else 1)
            lo, hi = (required - 1).bit_length(), 0
            if lo > (12 if ultra else 9):
                remainder_bits = (required - (1 << (lo - 1)) - 1).bit_length()
                if remainder_bits < lo - 1:
                    lo, hi = lo - 1, max(1, remainder_bits)
            if lo >= 32:
                return reject("fifo-capacity", "native memory size overflows unsigned int")
            capacity = (1 << lo) + ((1 << hi) if hi else 0) + (17 if ultra else 2)
        return FifoStorage(effective, capacity)

    @constraint
    def capacity_supported(self) -> bool | Rejected:
        """The native memory the storage decomposes into is addressable."""
        _ = self.storage
        return True

    admission = ConstraintGroup(geometry_supported, capacity_supported)

    input = WordPort(name="input", endpoint=Endpoint.TARGET, bits=word_bits)
    output = WordPort(name="output", endpoint=Endpoint.INITIATOR, bits=word_bits)

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "DATA_WIDTH": self.word_bits,
            "DEPTH": self.depth,
            "RAM_STYLE": f'"{self.ram_style}"',
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),)


__all__ = ["FifoKernel", "FifoStorage"]
