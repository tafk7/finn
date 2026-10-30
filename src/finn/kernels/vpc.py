# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's ``vpc``: the same elements in the same order, another number a beat.

It regroups ``lanes_in`` elements a beat into ``lanes_out``, through vectors of
their least common multiple. It presents the sequences it is given on its two ports
(``arriving``, ``leaving``: the same elements of its tensor, ``input_tensor``, in the
same order), and derives its generics from
them: ``lanes_in`` and ``lanes_out`` their lanes, ``W`` the tensor's element. Words
are opaque on FinnLib's native pins. On a channel, ``vpc`` is a stage of the
channel's adapter (``finn.kernels.adapters``), which gives it the two sides of its
width conversion.

FinnLib's ``N`` is the vector the RTL converts independently, padding its last
beats when ``N`` is no multiple of a lane count (``vpc.sv``: ``1 + (N-1)/PI``
input beats a vector). Both sides of a width conversion present whole beats of
the same elements, so the stream is whole least-common-multiple vectors and no
beat is padded: ``N`` is their least common multiple, which the RTL normalizes to
the same converter as any multiple of it (``N0``). Padded vectors would serve a
side presenting padded beats, which no beat sequence (``finn.dataflow``) states.
"""

from __future__ import annotations

from math import lcm
from typing import ClassVar

from finn.core.space import Param, derived, view
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING
from finn.kernels.library import ModuleSpec
from finn.kernels.port_model import StreamPort
from finn.kernels.utilization import Fit
from finn.platform.capabilities import Resources


def vpc_resources(*, w: int, pi: int, po: int) -> Resources:
    """FinnLib ``vpc``: registers for the PI + PO elements of W bits it holds between its
    two sides, and its lane multiplexers; a ``Fit`` over the bits held."""
    held = (pi + po) * w
    return Resources(lut=_VPC_LUT.at(held), ff=_VPC_FF.at(held))


# Feature: the bits held, (PI + PO) * W.
_VPC_LUT = Fit(9.9, (0.022,))
_VPC_FF = Fit(6.8, (1.0,))


class VpcKernel(Kernel):
    """FinnLib ``vpc``: its tensor's elements in the order they arrive, at another lane
    count; see the module docstring."""

    id = "finnlib.vpc"
    clock_domain = NATIVE_CLOCKING

    # The tensor whose elements it regroups, a fact, and the two sides of the conversion.
    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived
    def output_tensor(self) -> Tensor:
        """What it moves its input to: the same tensor."""
        tensor: Tensor = self.input_tensor
        return tensor

    input = StreamPort(
        endpoint=Endpoint.TARGET,
        transport=NATIVE,
        tensor=input_tensor,
        sequence=arriving,
    )
    output = StreamPort(
        endpoint=Endpoint.INITIATOR,
        transport=NATIVE,
        tensor=output_tensor,
        sequence=leaving,
    )

    @derived
    def element_bits(self) -> int:
        """W: the bits of its tensor's element."""
        bits: int = self.input.element.bits
        return bits

    @derived
    def lanes_in(self) -> int:
        """PI: the elements a beat its input presents."""
        lanes: int = self.input.stream.sequence.form.lanes
        return lanes

    @derived
    def lanes_out(self) -> int:
        """PO: the elements a beat its output presents."""
        lanes: int = self.output.stream.sequence.form.lanes
        return lanes

    # Its lanes and element come from its ports, positive by construction: the RTL
    # refuses nothing they present, so it states no constraint of its own.

    @view
    def resource_use(self) -> Resources:
        return vpc_resources(w=self.element_bits, pi=self.lanes_in, po=self.lanes_out)

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "N": lcm(self.lanes_in, self.lanes_out),
            "PAD_ZEROS": 1,
            "PI": self.lanes_in,
            "PO": self.lanes_out,
            "RELAX_THROUGHPUT": 0,
            "W": self.element_bits,
        }

    module_spec: ClassVar[ModuleSpec] = ModuleSpec(
        "vpc", (CopiedSource("finnlib", "rtl/shape/vpc.sv", provides=("module:vpc",)),)
    )


__all__ = ["VpcKernel", "vpc_resources"]
