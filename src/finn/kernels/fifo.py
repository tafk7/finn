# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's opaque-word FIFO, with its native unpadded ready/valid pins.

Words have no numerical datatype. DEPTH is the requested capacity; the native
implementation may round its storage up and forces a shift FIFO for shallow
depths. Reset is synchronous, active-high, and discards pending words.
"""

from finn.kernels.artifacts.abi import Clock, Direction, Reset, Signal
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


class FifoKernel(Kernel):
    id = "finnlib.fifo"
    version = "1"

    word_bits = Param(int)
    depth = Param(int)

    @constraint(bits=word_bits, depth=depth)
    def geometry_supported(*, bits: int, depth: int) -> bool | Rejected:
        if bits < 1 or depth < 2 or max(bits, depth) > 0xFFFFFFFF:
            return reject("fifo-geometry", "word_bits must be positive and depth at least two")
        return True

    ram_style = Decision(str, values=("auto", "shift", "distributed", "block", "ultra"))

    @view(
        semantics=default_semantics(ModuleBuildRequirements),
        constraints=(geometry_supported,),
        bits=word_bits,
        depth=depth,
        ram=ram_style,
    )
    def build_requirements(
        *, bits: int, depth: int, ram: str
    ) -> ModuleBuildRequirements | Rejected:
        if bits < 1:
            return reject("fifo-interface", "word_bits must be positive")
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
                Signal("idat", Direction.IN, bits),
                Signal("ivld", Direction.IN, 1),
                Signal("irdy", Direction.OUT, 1),
                Signal("odat", Direction.OUT, bits),
                Signal("ovld", Direction.OUT, 1),
                Signal("ordy", Direction.IN, 1),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        return ModuleBuildRequirements(
            FifoKernel.id,
            FifoKernel.version,
            parameters,
            abi,
            (CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),),
        )


__all__ = ["FifoKernel"]
