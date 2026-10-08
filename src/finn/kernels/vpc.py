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
from finn.kernels.utilization import RESOURCES_SEMANTICS, Fit, Resources


def vpc_resources(*, w: int, pi: int, po: int) -> Resources:
    """FinnLib ``vpc``: registers for the PI + PO elements of W bits it holds between its
    two sides, and its lane multiplexers; a ``Fit`` over the bits held."""
    held = (pi + po) * w
    return Resources(lut=_VPC_LUT.at(held), ff=_VPC_FF.at(held))


# Feature: the bits held, (PI + PO) * W.
_VPC_LUT = Fit(9.9, (0.022,))
_VPC_FF = Fit(6.8, (1.0,))


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

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return vpc_resources(w=self.element_bits, pi=self.lanes_in, po=self.lanes_out)

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


__all__ = ["VpcKernel", "vpc_resources"]
