# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's ``vpc``: the same elements in the same order, another number a beat.

It regroups ``lanes_in`` elements a beat into ``lanes_out``, through vectors of
their least common multiple. Words are opaque on FinnLib's native pins. On a
channel, ``vpc`` is a stage of the channel's adapter (``finn.kernels.adapters``).
"""

from __future__ import annotations

from collections.abc import Mapping
from math import lcm

from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.port import WordPort


class VpcKernel(Kernel):
    id = "finnlib.vpc"
    version = 1
    rtl_module = "vpc"

    element_bits: int = Param()
    lanes_in: int = Param()
    lanes_out: int = Param()

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        if min(self.element_bits, self.lanes_in, self.lanes_out) < 1:
            return reject("vpc-geometry", "element width and lane counts must be positive")
        return True

    admission = ConstraintGroup(geometry_supported)

    input = WordPort(name="input", endpoint=Endpoint.TARGET, bits=lanes_in * element_bits)
    output = WordPort(name="output", endpoint=Endpoint.INITIATOR, bits=lanes_out * element_bits)

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "N": lcm(self.lanes_in, self.lanes_out),
            "PAD_ZEROS": 1,
            "PI": self.lanes_in,
            "PO": self.lanes_out,
            "RELAX_THROUGHPUT": 0,
            "W": self.element_bits,
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/shape/vpc.sv", provides=("module:vpc",)),)


__all__ = ["VpcKernel"]
