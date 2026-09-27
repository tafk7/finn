# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's opaque-word FIFO, with its native unpadded ready/valid pins.

Words have no numerical datatype. DEPTH is the requested capacity; the native
implementation may round its storage up and forces a shift FIFO for shallow
depths. Reset is synchronous, active-high, and discards pending words.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
from finn.kernels.physical.stream import STREAM_INTERFACES, ReadyValidStream
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    default_semantics,
    reject,
    view,
)


@dataclass(frozen=True)
class FifoStorage:
    """Native storage selection and capacity; synthesis resource inference is separate."""

    requested_style: str
    effective_style: str
    requested_depth: int
    capacity: int


class FifoKernel(Kernel):
    id = "finnlib.fifo"
    version = "1"

    word_bits: int = Param()
    depth: int = Param()

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        bits = self.word_bits
        depth = self.depth
        if bits < 1 or depth < 2 or max(bits, depth) > 0xFFFFFFFF:
            return reject("fifo-geometry", "word_bits must be positive and depth at least two")
        return True

    ram_style: str = Decision(values=("auto", "shift", "distributed", "block", "ultra"))

    @view(semantics=default_semantics(FifoStorage), requires=(geometry_supported,))
    def storage(self) -> FifoStorage | Rejected:
        depth, style = self.depth, self.ram_style
        if not 2 <= depth <= 0xFFFFFFFF:
            return reject("fifo-geometry", "depth must fit native unsigned int and be at least two")
        effective = (
            "shift"
            if depth <= 33 or style == "distributed"
            else style
            if style != "auto"
            else "shift"
            if depth <= 64
            else "block"
            if depth <= 2028
            else "ultra"
        )
        if effective == "shift":
            capacity = max(5, depth)
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
        return FifoStorage(style, effective, depth, capacity)

    @view(semantics=STREAM_INTERFACES)
    def interfaces(self) -> tuple[ReadyValidStream, ...] | Rejected:
        bits = self.word_bits
        if not 1 <= bits <= 0xFFFFFFFF:
            return reject(
                "fifo-interface", "word_bits must be positive and fit native unsigned int"
            )
        return (
            ReadyValidStream("input", bits, Endpoint.TARGET, "idat", "ivld", "irdy", "clk", "rst"),
            ReadyValidStream(
                "output", bits, Endpoint.INITIATOR, "odat", "ovld", "ordy", "clk", "rst"
            ),
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(geometry_supported,))
    def build_requirements(self) -> ModuleBuildRequirements | Rejected:
        bits = self.word_bits
        depth = self.depth
        ram = self.storage().requested_style
        streams = self.interfaces()
        parameters = (("DATA_WIDTH", bits), ("DEPTH", depth), ("RAM_STYLE", f'"{ram}"'))
        abi = ModuleABIRequirements(
            FixedModuleName("fifo"),
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
            FifoKernel.id,
            FifoKernel.version,
            parameters,
            abi,
            (CopiedSource("finnlib", "rtl/fifo.sv", provides=("module:fifo",)),),
        )


__all__ = ["FifoKernel", "FifoStorage"]
