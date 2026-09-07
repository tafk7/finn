# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One weight tile, several activation rows: a third Network for the same node.

```text
activation boundary -> compute -> output boundary
        weight boundary --^
```

The other two Designs replay the activation so that each weight tile is used
once per row.  This one turns that around: the tile is fetched once and reused
across ``interleave`` consecutive rows, so the weight stream carries
``PE * SIMD / interleave`` elements per beat and the activation is presented
once, uncompacted and unreplayed.

**It places one node, and that is the finding rather than a simplification.**
There is no replay Region to compose with, because interleaving removes the
thing replay existed to supply.  A Design is a composition of KernelChoice, not
necessarily of *several* KernelChoice, and the shape of the composition follows the
arithmetic rather than the other way round.

**``interleave`` is an ordinary Design-owned Decision.**  It is in the compute
Region's dependency closure -- it changes the schedule, the weight beat
sequence, and therefore the Network's weight boundary -- so it cannot be a
Kernel-local Decision, by the same rule that puts PE and SIMD here.  It needs no
new persistence: the recorded document names
``design.batch_interleaved.interleave`` because the walk found it, not because
anything here declared how to store it.
"""

from __future__ import annotations

from finn.dataflow.designs.design import NetworkBoundary, KernelChoice
from finn.dataflow.kernels.dotp_axi import BatchInterleavedDotpAxiKernel
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Subspace,
    constraint,
    divisors_of,
    reject,
)
from finn.dataflow.ops.mvau.designs.base import SHARED_INPUTS, WeightedDotProductDesign


class BatchInterleavedDesign(WeightedDotProductDesign):
    """The dot product with its weight tile amortized across a batch of rows."""

    id = "batch_interleaved"
    version = "1"

    repetitions = WeightedDotProductDesign.repetitions
    matrix_width = WeightedDotProductDesign.matrix_width
    matrix_height = WeightedDotProductDesign.matrix_height
    activation_type = WeightedDotProductDesign.activation_type
    weight_type = WeightedDotProductDesign.weight_type
    accumulator_type = WeightedDotProductDesign.accumulator_type
    output_type = WeightedDotProductDesign.output_type
    narrow_weights = WeightedDotProductDesign.narrow_weights
    target_dsp = WeightedDotProductDesign.target_dsp
    clock_period_ns = WeightedDotProductDesign.clock_period_ns
    pe = WeightedDotProductDesign.pe
    simd = WeightedDotProductDesign.simd

    #: How many consecutive rows share one weight tile.  A divisor of the row
    #: count, because a batch that did not divide evenly would leave a partial
    #: final batch that the Region has no schedule level for.
    interleave = Decision(int, domain=divisors_of(repetitions))

    compute = KernelChoice(
        Subspace(
            BatchInterleavedDotpAxiKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
            interleave=interleave,
        ),
    )

    activation = NetworkBoundary(compute.input("activation"))
    weight = NetworkBoundary(compute.input("weight"))
    output = NetworkBoundary(compute.output("output"))

    @constraint(interleave=interleave)
    def interleaving_is_more_than_one(*, interleave: int) -> object:
        """At one, this Design *is* the streamed one, and says so rather than tying.

        Two points that build the same thing are worse than one point fewer:
        a chooser would have no way to prefer either, and evidence gathered
        against one would silently describe the other.
        """

        if interleave > 1:
            return True
        return reject(
            "mvau-interleave-degenerate",
            "an interleave of one is exactly the streamed dot product; use that Design",
            values={"interleave": interleave},
        )

    @constraint(interleave=interleave, pe=pe, simd=simd)
    def interleaving_divides_the_weight_tile(*, interleave: int, pe: int, simd: int) -> object:
        """The tile is split into equal chunks, so the split has to be exact.

        Stated as a constraint as well as enforced in the Region constructor:
        the constructor's refusal keeps a malformed Region from existing, and
        this one makes the reason a *verdict* a caller can read off the
        assessment rather than an unresolved Network.
        """

        if (pe * simd) % interleave == 0:
            return True
        return reject(
            "mvau-interleave-uneven-tile",
            f"an interleave of {interleave} does not divide the {pe * simd}-element "
            "weight tile into equal chunks",
            values={"interleave": interleave, "tile": pe * simd},
        )

    #: The base class's constraint is carried forward explicitly; overriding the
    #: group without naming it would silently drop it.
    dataflow_support = ConstraintGroup(
        *WeightedDotProductDesign.dataflow_support.constraints,
        interleaving_is_more_than_one,
        interleaving_divides_the_weight_tile,
        name="interleave_available",
    )


#: Every Input the Design consumes, for a caller assembling the bindings.
DESIGN_INPUTS = SHARED_INPUTS

__all__ = ["DESIGN_INPUTS", "BatchInterleavedDesign"]
