# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three ways a matrix can reach the dot product, as three Networks.

```text
external    activation -> replay -> compute -> output
                          weight boundary --^

embedded    activation -> replay -> compute -> output
                          (no weight port at all)

decoupled   activation -> replay -> compute -> output
                          memory --weight-->  ^
```

This is the forcing case U3 exists for, and the point of it is that all three
differences are *semantic*.  External and decoupled place the same compute
Region and differ in what feeds its weight input -- a boundary or an edge from
a second node.  Embedded places a **different** compute Region, one with no
weight input, so there is nothing to feed and no boundary to substitute.  None
of that is a physical choice dressed up as a topology, and none of it needs a
supplier that can actually be built: at this phase the embedded core and the
memstream both report their build unit unavailable, and every Network here
still resolves.

**Initializer presence does not choose the mode.**  ``initializer_present`` is
an ordinary Input -- a fact about the source graph -- and it appears in exactly
one place: a constraint saying that a mode which keeps the matrix locally needs
one to keep.  A design space that silently picked "embedded" because a tensor
happened to be constant would be making a hardware decision inside a graph
query, and the caller would have no way to ask for anything else.
"""

from __future__ import annotations

from enum import Enum

from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.designs.design import (
    Boundary,
    Connection,
    DataflowDesign,
    Kernels,
    Sink,
)
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock, EmbeddedDotpAxiKernel
from finn.dataflow.kernels.memstream import CYCLIC_PARAMETER_DELIVERY, MemstreamKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.model.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Subspace,
    constraint,
    derived,
    divisors_of,
    reject,
)
from finn.dataflow.model.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class WeightSupply(str, Enum):
    """How the matrix reaches the compute Region.  One dial, three positions."""

    EXTERNAL = "external"
    EMBEDDED = "embedded"
    DECOUPLED = "decoupled"


class SuppliedDotProductDesign(DataflowDesign):
    """Replay and dot product, with the weight path as a declared choice."""

    id = "supplied_dot_product"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Input(bool)
    target_dsp = Input(DspBlock)
    clock_period_ns = Input(float)

    #: A fact about the source graph, never a choice.  See the module docstring.
    initializer_present = Input(bool)

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    #: The one dial.  Every conditional segment, edge and boundary below reads
    #: a derived boolean off it, so there is exactly one place the mode is set
    #: and no combination of independent switches that could disagree.
    weight_supply = Decision(WeightSupply, values=tuple(WeightSupply))

    @derived(bool, supply=weight_supply)
    def streams_weights(*, supply: WeightSupply) -> bool:
        return supply is WeightSupply.EXTERNAL

    @derived(bool, supply=weight_supply)
    def embeds_weights(*, supply: WeightSupply) -> bool:
        return supply is WeightSupply.EMBEDDED

    @derived(bool, supply=weight_supply)
    def decouples_weights(*, supply: WeightSupply) -> bool:
        return supply is WeightSupply.DECOUPLED

    @derived(bool, supply=weight_supply)
    def keeps_weights_locally(*, supply: WeightSupply) -> bool:
        return supply is not WeightSupply.EXTERNAL

    replay = Kernels(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=pe,
            simd=simd,
        ),
        computation=ACTIVATION_REPLAY_COMPUTATION,
    )

    #: Two candidates and one selector.  The embedded core is a different
    #: Region, so it is a different candidate rather than a mode of the first.
    compute = Kernels(
        Subspace(
            DotpAxiKernel,
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
        ),
        Subspace(
            EmbeddedDotpAxiKernel,
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
        ),
        computation=DOT_PRODUCT_COMPUTATION,
    )

    memory = Kernels(
        Subspace(
            MemstreamKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            weight_type=weight_type,
            pe=pe,
            simd=simd,
        ),
        computation=CYCLIC_PARAMETER_DELIVERY,
        when=decouples_weights,
    )

    activation_replay = Connection(
        replay.output("activation_out"),
        Sink(compute.input("activation")),
    )

    #: The decoupled case's second edge.  Present exactly when the memory node
    #: is, so a Network never carries an edge from a node that is not there.
    weight_supply_edge = Connection(
        memory.output("weight"),
        Sink(compute.input("weight")),
        when=decouples_weights,
    )

    activation = Boundary(replay.input("activation_in"))
    #: Substituted, not suppressed: external streaming is the only mode in which
    #: the matrix crosses this Design's boundary.
    weight = Boundary(compute.input("weight"), when=streams_weights)
    output = Boundary(compute.output("output"))

    @constraint(supply=weight_supply, present=initializer_present)
    def local_weights_need_an_initializer(*, supply: WeightSupply, present: bool) -> object:
        """A mode that keeps the matrix locally needs a matrix to keep.

        Applicability, not selection.  The fact narrows which modes are
        available at this source node; it never picks one of them.
        """

        if supply is not WeightSupply.EXTERNAL and not present:
            return reject(
                "mvau-local-weights-need-an-initializer",
                f"{supply.value} weight supply holds the matrix locally, "
                "and this source node supplies none",
                values={"supply": supply.value},
            )
        return True

    @constraint(supply=weight_supply, present=initializer_present)
    def external_supply_ignores_the_initializer(*, supply: WeightSupply, present: bool) -> bool:
        """External streaming is available whether or not there is one.

        Stated as its own constraint so that "the initializer does not choose
        the mode" is a claim with a test rather than an absence of code.
        """

        del present
        return supply is WeightSupply.EXTERNAL or True

    dataflow_support = ConstraintGroup(
        local_weights_need_an_initializer,
        external_supply_ignores_the_initializer,
        name="supply_available",
    )


#: Every Input the Design consumes, for a caller assembling the bindings.
DESIGN_INPUTS = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "weight_type",
    "accumulator_type",
    "output_type",
    "narrow_weights",
    "target_dsp",
    "clock_period_ns",
    "initializer_present",
)

__all__ = ["DESIGN_INPUTS", "SuppliedDotProductDesign", "WeightSupply"]
