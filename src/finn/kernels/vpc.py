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

from finn.core.space import Param, Rejected, derived, view
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Clocking, Endpoint, Pin
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import Kernel
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.port import held, pins
from finn.kernels.port_model import SequencePort
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

    @derived
    def port_clocking(self) -> Clocking:
        """Physical port clock names remain the class's clock_domain, including overrides."""
        return type(self).clock_domain

    input = SequencePort(
        label="input",
        endpoint=Endpoint.TARGET,
        transport=NATIVE,
        tensor=input_tensor,
        sequence=arriving,
        clocking=port_clocking,
    )
    output = SequencePort(
        label="output",
        endpoint=Endpoint.INITIATOR,
        transport=NATIVE,
        tensor=output_tensor,
        sequence=leaving,
        clocking=port_clocking,
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

    def _input_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.input.presenting()
        return pins(
            self.input.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.input.clocking,
        )

    def _output_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.output.presenting()
        return pins(
            self.output.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.output.clocking,
        )

    def _input_held(self) -> Held | Rejected:
        if self.input.placed:
            return Held()
        element = self.input.element
        return held(self.input.physical_spec(idle=True), element, self.input.clocking)

    def _output_held(self) -> Held | Rejected:
        if self.output.placed:
            return Held()
        element = self.output.element
        return held(self.output.physical_spec(idle=True), element, self.output.clocking)

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pin_parts = _each((self._input_pins(), self._output_pins()))
        if isinstance(pin_parts, Rejected):
            return pin_parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        assert sources is not None  # Vpc has constant sources; no dynamic source View.
        refused = module_unprovided(spec, type(self).__qualname__, sources)
        if isinstance(refused, Rejected):
            return refused
        data = ()
        held_parts = _each((self._input_held(), self._output_held()))
        if isinstance(held_parts, Rejected):
            return held_parts
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            tuple(pin_parts),
            sources,
            data,
            tuple(held_parts),
            (True, True),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return max(self.input.stream.sequence.form.beats, self.output.stream.sequence.form.beats)


__all__ = ["VpcKernel", "vpc_resources"]
