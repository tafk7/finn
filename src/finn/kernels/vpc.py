# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's ``vpc``: the same elements in the same order, another number a beat.

It regroups ``lanes_in`` elements a beat into ``lanes_out``, through vectors of
their least common multiple. It presents the sequences it is given on its two ports
(``arriving``, ``leaving``), and derives its generics from
them: ``lanes_in`` and ``lanes_out`` their lanes, ``W`` the tensor's element. The two
are one sequence at two lane counts, its association: the same elements of its tensor,
``input_tensor``, in the same order, repeated alike, with the same markers. It refuses
any other pair (``width_only``, ``vpc-width-only``): its RTL only regroups lanes, and
where the two lane counts are the same, its wire branch passes each beat through. Words
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

from math import gcd, lcm
from typing import ClassVar

from finn.core.space import Param, Rejected, constraint, derived, reject, view
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, regrouped
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING, Intake, PortIntake, Standing
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
    # refuses nothing they present. The kernel refuses a pair of sequences the RTL does
    # not associate.

    @constraint
    def width_only(self) -> bool | Rejected:
        """Its association: ``leaving`` is ``arriving`` at another lane count, or the same.
        The two present one element sequence (``regrouped`` at one lane: the order whose
        equality ``classify`` names a width conversion, which a lane group has too), repeat
        their pass alike and carry the same markers. Any other pair is a step its RTL does
        not take: a reorder or a lane permutation (its wire branch passes each beat through
        as it arrives), or a repetition or a marker it would make or drop."""
        arriving: BeatSequence = self.arriving
        leaving: BeatSequence = self.leaving
        if regrouped(arriving.form, 1) != regrouped(leaving.form, 1):
            return reject(
                "vpc-width-only",
                "leaving presents its elements in another order than arriving: "
                "a vpc only regroups lanes",
            )
        if arriving.repetition is not leaving.repetition:
            return reject(
                "vpc-width-only",
                f"arriving repeats {arriving.repetition.value} and leaving "
                f"{leaving.repetition.value}: a vpc repeats no pass",
            )
        if arriving.markers != leaving.markers:
            return reject(
                "vpc-width-only",
                "arriving and leaving carry different markers: a vpc makes and drops none",
            )
        return True

    @derived
    def intake(self) -> Intake:
        """By the branch its lanes select (``vpc.sv``, its lanes normalized by their GCD,
        ``PI0`` and ``PO0``; ``N0 = PI0 * PO0``, whole vectors; ``RELAX_THROUGHPUT`` 0),
        from reset with its output stalled, in input beats (CONTRACT.md § 5):

        - the wire (``PI0 == PO0 == 1``) takes nothing ahead: its ready is its output's;
        - the serializer (``PO0 == 1``) takes one beat into its buffer, its first element
          sidestepped, and stops until the buffer drains: at most that beat and one element
          of the one before;
        - the deserializer (``PI0 == 1``) takes ``PO0`` beats into its buffer and one more
          into its sidestep;
        - the general branch takes a beat while it holds at most ``PO0`` elements (its
          capacity counter): ``PO0 // PI0 + 1`` beats from reset, and holds at most
          ``PI0 + PO0`` elements, the beats they span ``1 + ceil(PO0 / PI0)``.
        """
        common = gcd(self.lanes_in, self.lanes_out)
        pi0, po0 = self.lanes_in // common, self.lanes_out // common
        if pi0 == po0 == 1:
            counts = (0, 0)
            rule = "takes a beat only as its output takes one (ready passes through)"
            source = "rtl/shape/vpc.sv:78-86 (genWire)"
        elif po0 == 1:
            counts = (1, 2)
            rule = (
                "takes a beat into an empty buffer, its first element into the sidestep; "
                "resumes once the buffer has shifted out"
            )
            source = "rtl/shape/vpc.sv:118-168 (genSer, genFull: Cnt, irdy :163)"
        elif pi0 == 1:
            counts = (po0 + 1, po0 + 1)
            rule = (
                "takes beats until its buffer is full and one into its sidestep; resumes "
                "when the output takes the buffer"
            )
            source = "rtl/shape/vpc.sv:202-252 (genDes, genFull: Cnt, SRdy :249)"
        else:
            counts = (po0 // pi0 + 1, 1 + -(-po0 // pi0))
            rule = (
                f"takes a beat while it holds at most {po0} of the {pi0 + po0} words it has "
                "room for (a word: the lanes' GCD of elements); resumes as the output takes "
                "words"
            )
            source = "rtl/shape/vpc.sv:261-347 (genGeneric: CAP :266, ICap :316-322, irdy :325)"
        before, most = counts
        return Intake((PortIntake("input", before, most, rule, Standing.STRUCTURAL, source),))

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
